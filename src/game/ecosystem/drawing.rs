//! Which creatures are drawn each frame, posed and smoothed between the simulation's steps, and
//! the census of who lives near the player.

use crate::{
    game::{
        ecosystem::{
            EcologicalMotion, Ecosystem, Herd, Individual, Physiology, Relations, STEP, Species,
        },
        flight::{Mount, Player},
    },
    math::ToScene,
};
use bevy::prelude::*;
use rayon::prelude::*;
use skybound::scene::life::{
    CLUTCH, Drawn, FULL_POUCH, HERD, KINDS, Kind, Plant,
    animalia::physiology::{Brood, plant},
    behaviour::Activity,
    development,
    genome::Genome,
    vitals::Vitals,
};
use spirv_std::glam::{Quat as SceneQuat, Vec3 as SceneVec3, Vec4 as SceneVec4};
use std::{collections::HashMap, fmt::Write};

pub(super) type Seen<'a> = (
    Entity,
    &'a Species,
    &'a Physiology,
    &'a Transform,
    &'a EcologicalMotion,
    Option<&'a Individual>,
    &'a Relations,
);

/// Every minute, how many of each species are alive, so a lineage dying out shows in the log.
pub(super) fn census(
    organisms: Query<(&Species, &Physiology, &Relations)>,
    ecosystem: Res<Ecosystem>,
    time: Res<Time>,
    mut last: Local<f32>,
) {
    if time.elapsed_secs() - *last < 60.0 {
        return;
    }
    *last = time.elapsed_secs();
    let mut counts: HashMap<Kind, usize> = HashMap::new();
    let mut doing: HashMap<String, usize> = HashMap::new();
    for (species, body, relations) in &organisms {
        if body.0.health > 0.0 {
            *counts.entry(species.0).or_default() += 1;
            if !plant(species.0) {
                *doing
                    .entry(format!("{:?}", relations.activity))
                    .or_default() += 1;
            }
        }
    }
    let mut line = String::new();
    for kind in KINDS {
        let _ = write!(line, " {kind:?}={}", counts.get(kind).copied().unwrap_or(0));
    }
    info!(
        "census{line} births={} deaths={}",
        ecosystem.births, ecosystem.deaths
    );
    let mut activities: Vec<_> = doing.into_iter().collect();
    activities.sort();
    info!("activity {activities:?}");
}

/// Which parents carry young, in what and how many, by entity; young drawn as part of their parent
/// are marked zero.
pub(super) fn carried(
    organisms: &Query<Seen, (Without<Player>, Without<Mount>)>,
) -> HashMap<Entity, (u32, u32)> {
    let mut carrying: HashMap<Entity, (u32, u32)> = HashMap::new();
    for (entity, species, body, _, _, _, relations) in organisms {
        let biology = species.0.biology();
        let young = body.0.age_days / f64::from(biology.maturity_days);
        if relations.host.is_some() && relations.host == relations.parent {
            let brood = match species.0.animal().map(|animal| animal.brood) {
                Some(Brood::Carried) if young < 0.3 => CLUTCH,
                Some(Brood::Pouch) => FULL_POUCH,
                _ => continue,
            };
            let held = carrying
                .entry(relations.host.unwrap_or(entity))
                .or_insert((brood, 0));
            held.1 += 1;
            // Eggs glued to the parent are drawn as part of it; pouch young are drawn themselves,
            // sitting in the pouch.
            if brood == CLUTCH {
                carrying.insert(entity, (0, 0));
            }
        }
    }
    carrying
}

/// Pose the creatures that will cover the most of the screen this frame, carrying each one's last
/// pose along for motion vectors.
pub(super) fn gather_herd(
    mut herd: ResMut<Herd>,
    mut last: Local<HashMap<Entity, Drawn>>,
    camera: Single<&GlobalTransform, With<Camera3d>>,
    organisms: Query<Seen, (Without<Player>, Without<Mount>)>,
    time: Res<Time>,
) {
    let eye = camera.translation();
    let forward = camera.forward();
    let carrying = carried(&organisms);
    let mut ranked: Vec<(f32, Entity)> = organisms
        .iter()
        .filter_map(|(entity, species, _, transform, _, _, _)| {
            if carrying.get(&entity).is_some_and(|held| held.0 == 0) {
                return None;
            }
            let radius = species.0.render_radius();
            let offset = transform.translation - eye;
            let distance = offset.length().max(1.0);
            // Behind the camera, or too small to cover a pixel.
            if offset.dot(*forward) < -radius || radius / distance < 0.0015 {
                return None;
            }
            Some((radius / distance, entity))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(HERD);
    let now = time.elapsed_secs();
    // What each pose needs, read out of the world so the poses, each with its own tentacle and
    // cloth simulation, can be built across every core.
    let chosen: Vec<_> = ranked
        .into_iter()
        .filter_map(|(_, entity)| {
            let (_, species, body, transform, motion, individual, relations) =
                organisms.get(entity).ok()?;
            // Drawn partway from where the last step found it to where it left it, so it moves
            // smoothly between the simulation's steps.
            let along = ((now - relations.stepped) / STEP).clamp(0.0, 1.0);
            let transform = Transform {
                translation: transform.translation + relations.lag * (1.0 - along),
                rotation: relations.turned.slerp(transform.rotation, along),
                ..*transform
            };
            Some(Shown {
                entity,
                kind: species.0,
                age: body.0.age_days as f32,
                transform,
                // A displaying animal holds still, its crest raised.
                speed: if relations.activity == Activity::Display {
                    0.0
                } else {
                    motion.0.length()
                },
                seed: individual.map_or_else(|| entity.to_bits() as u32, |identity| identity.0),
                activity: relations.activity,
                tuck: relations.senses.goal.map_or(0.0, |goal| goal.fold),
                held: carrying.get(&entity).copied().unwrap_or((0, 0)),
                genes: relations.genes,
                vitals: body.0,
                wingbeat: relations.wingbeat,
                effort: relations.effort,
                bend: relations.bend,
                blunted: relations.mind.wear,
                tether: relations
                    .tether
                    .and_then(|(anchor, _)| organisms.get(anchor).ok())
                    .map(|(_, _, _, anchor, ..)| anchor.translation - transform.translation),
            })
        })
        .collect();
    let poses: Vec<(Entity, Drawn)> = chosen
        .par_iter()
        .map(|shown| (shown.entity, pose(shown, now)))
        .collect();
    let mut drawn = HashMap::with_capacity(poses.len());
    herd.0.clear();
    for (entity, mut pose) in poses {
        if let Some(old) = last.get(&entity)
            && old
                .organism()
                .position
                .truncate()
                .distance(pose.organism().position.truncate())
                < 100.0
        {
            pose.follow(old);
        }
        drawn.insert(entity, pose);
        herd.0.push(pose);
    }
    *last = drawn;
}

/// Draw the silk tying a plant to another, `offset` away: its far end in the plant's own frame, a
/// cable as thick as a Weaver lays, and bounds grown to hold it.
fn tie(plant: &mut Plant, offset: Vec3) {
    let c = &mut plant.organism;
    let turn = SceneQuat::from_array(c.rotation.to_array()).conjugate();
    let scale = development::scale(c);
    let end = (turn * offset.to_scene()) / scale;
    let thick = 0.3 / scale.min_element();
    plant.tether = end.extend(thick);
    let c = &mut plant.organism;
    c.opaque_bounds = c
        .opaque_bounds
        .truncate()
        .max(end.abs() + SceneVec3::splat(thick * 2.0))
        .extend(c.opaque_bounds.w);
    c.position.w = c.position.w.max(offset.length());
}

/// What drawing one organism needs, read out of the world.
struct Shown {
    entity: Entity,
    kind: Kind,
    age: f32,
    transform: Transform,
    speed: f32,
    seed: u32,
    activity: Activity,
    tuck: f32,
    held: (u32, u32),
    genes: Genome,
    vitals: Vitals,
    wingbeat: f32,
    effort: f32,
    bend: Vec3,
    blunted: f32,
    /// How far off the plant it is tied to by silk lies, if it is tied.
    tether: Option<Vec3>,
}

/// An organism's pose for the renderer, from what the simulation did to it: its wings where their
/// beat has reached and spread as wide as it is working them, a plant bent by the air, and the
/// tears, scars and wear its life has left on it.
fn pose(shown: &Shown, now: f32) -> Drawn {
    let Shown {
        kind,
        age,
        transform,
        speed,
        seed,
        activity,
        tuck,
        held,
        genes,
        vitals,
        wingbeat,
        effort,
        bend,
        blunted,
        tether,
        ..
    } = *shown;
    // Resting animals fold their wings and stop beating them.
    let resting = matches!(
        activity,
        Activity::Rest | Activity::Attach | Activity::Sleep
    );
    let fold = if resting { 0.9 } else { tuck };
    let stroke = wingbeat.sin() * (0.05 + effort.min(1.0) * 0.35) * (1.0 - fold);
    let mut pose = Drawn::born(
        transform
            .translation
            .extend(kind.render_radius())
            .to_scene(),
        transform.rotation.to_array().into(),
        Vec4::new(stroke, fold, now, speed).to_scene(),
        kind as u32,
        age,
        seed,
        genes,
    );
    // Jaws wide as it bites, bared at a hunter it drives off, parted as it strikes or holds out
    // food for a guardian, shut otherwise.
    let gape = match activity {
        Activity::Feed => 1.0,
        Activity::Defend => 0.6,
        Activity::Strike => 0.4,
        Activity::Offer => 0.3,
        _ => 0.0,
    };
    pose.organism_mut().wounds = vitals.wounds;
    if let Drawn::Plant(plant) = &mut pose {
        plant.bend = bend.to_scene().extend(1.0);
        plant.previous_bend = plant.bend;
        plant.water = SceneVec4::new(vitals.hydration, 0.0, 0.0, 0.0);
        if let Some(offset) = tether {
            tie(plant, offset);
        }
    }
    if let Some(animal) = pose.animal_mut() {
        animal.motion = SceneVec4::new(wingbeat, effort.min(1.0), gape, 1.0);
        animal.previous_motion = animal.motion;
        animal.hide = SceneVec4::new(vitals.overgrowth, blunted, 0.0, 0.0);
        animal.stores = SceneVec4::new(vitals.iron, 0.0, 0.0, held.1 as f32);
        animal.sex = SceneVec4::new(vitals.sex(kind), 0.0, 0.0, 0.0);
    }
    if let Some(wings) = pose.wings_mut() {
        wings.torn = SceneVec4::from_array(vitals.torn);
        wings.scarred = SceneVec4::from_array(vitals.scarred);
    }
    pose.carrying(vitals.expecting, vitals.gestation);
    if held.0 != 0 {
        pose.organism_mut().kind.z = held.0;
    }
    pose
}
