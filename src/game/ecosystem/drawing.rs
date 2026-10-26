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
    CLUTCH, CreatureInstance, FULL_POUCH, HERD, KINDS, Kind,
    animalia::physiology::{Brood, plant},
    behaviour::Activity,
};
use std::{collections::HashMap, fmt::Write};

pub(super) type Drawn<'a> = (
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

/// Which parents carry young, and in what, by entity; young drawn as part of their parent are
/// marked zero.
pub(super) fn carried(
    organisms: &Query<Drawn, (Without<Player>, Without<Mount>)>,
) -> HashMap<Entity, u32> {
    let mut carrying: HashMap<Entity, u32> = HashMap::new();
    for (entity, species, body, _, _, _, relations) in organisms {
        let biology = species.0.biology();
        let young = body.0.age_days / f64::from(biology.maturity_days);
        if relations.host.is_some() && relations.host == relations.parent {
            let brood = match species.0.animal().map(|animal| animal.brood) {
                Some(Brood::Carried) if young < 0.3 => CLUTCH,
                Some(Brood::Pouch) => FULL_POUCH,
                _ => continue,
            };
            carrying.insert(relations.host.unwrap_or(entity), brood);
            // Eggs glued to the parent are drawn as part of it; pouch young are drawn themselves,
            // sitting in the pouch.
            if brood == CLUTCH {
                carrying.insert(entity, 0);
            }
        }
    }
    carrying
}

/// Pose the creatures that will cover the most of the screen this frame, carrying each one's last
/// pose along for motion vectors.
pub(super) fn gather_herd(
    mut herd: ResMut<Herd>,
    mut last: Local<HashMap<Entity, CreatureInstance>>,
    camera: Single<&GlobalTransform, With<Camera3d>>,
    organisms: Query<Drawn, (Without<Player>, Without<Mount>)>,
    time: Res<Time>,
) {
    let eye = camera.translation();
    let forward = camera.forward();
    let carrying = carried(&organisms);
    let mut ranked: Vec<(f32, Entity)> = organisms
        .iter()
        .filter_map(|(entity, species, _, transform, _, _, _)| {
            if carrying.get(&entity) == Some(&0) {
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
            Some((
                entity,
                species.0,
                body.0.age_days as f32,
                transform,
                // A displaying animal holds still, its crest raised.
                if relations.activity == Activity::Display {
                    0.0
                } else {
                    motion.0.length()
                },
                individual.map_or_else(|| entity.to_bits() as u32, |identity| identity.0),
                relations.activity,
                relations.senses.goal.map_or(0.0, |goal| goal.fold),
                carrying.get(&entity).copied().unwrap_or(0),
            ))
        })
        .collect();
    let poses: Vec<(Entity, CreatureInstance)> = chosen
        .par_iter()
        .map(
            |&(entity, kind, age, transform, speed, seed, activity, tuck, held)| {
                // Each individual beats its wings at its own phase and pace, harder the faster
                // it flies.
                let phase = (seed % 997) as f32 * 0.37;
                let rate = 3.0 + (seed % 13) as f32 * 0.12;
                // Resting animals fold their wings and stop beating them.
                let resting = matches!(
                    activity,
                    Activity::Rest | Activity::Attach | Activity::Sleep
                );
                let fold = if resting { 0.9 } else { tuck };
                let stroke = ((now + phase) * rate).sin()
                    * (0.08 + (speed / 40.0).min(1.0) * 0.25)
                    * (1.0 - fold);
                let mut pose = CreatureInstance::specimen(
                    transform
                        .translation
                        .extend(kind.render_radius())
                        .to_scene(),
                    transform.rotation.to_array().into(),
                    Vec4::new(stroke, fold, now + phase, speed).to_scene(),
                    kind as u32,
                    age,
                    seed,
                );
                if held != 0 {
                    pose.kind.z = held;
                }
                (entity, pose)
            },
        )
        .collect();
    let mut drawn = HashMap::with_capacity(poses.len());
    herd.0.clear();
    for (entity, mut pose) in poses {
        if let Some(old) = last.get(&entity)
            && old.position.truncate().distance(pose.position.truncate()) < 100.0
        {
            pose.previous_position = old.position;
            pose.previous_rotation = old.rotation;
            pose.previous_animation = old.animation;
            pose.previous_proportions = old.proportions;
            pose.previous_tail = old.tail;
            pose.previous_cloth = old.cloth;
        }
        drawn.insert(entity, pose);
        herd.0.push(pose);
    }
    *last = drawn;
}
