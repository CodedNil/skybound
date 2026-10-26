//! What an organism senses and wants: its neighbours, the air, its body's drives, and what its
//! mind makes of them.

use crate::{
    game::{
        ecosystem::{
            Agent, Call, Relations,
            breeding::{Stage, nurture},
        },
        world::WorldData,
    },
    math::ToScene,
};
use bevy::prelude::*;
use skybound::scene::{
    climate::Climate,
    life::{
        Kind,
        animalia::{
            mind::{Drives, Glimpse, Mind, Perception, Sighting},
            physiology::{Take, plant},
        },
        behaviour::{Activity, Goal, Senses},
        organs::Function,
    },
};
use spirv_std::glam::Vec3 as SceneVec3;
use std::collections::HashMap;

pub(super) fn perceive(
    agent: Agent,
    relations: &mut Relations,
    climate: Climate,
    neighbours: &[Agent],
    now: f32,
    world: &WorldData,
) -> Senses {
    let Agent {
        entity,
        kind,
        vitals: body,
        position,
        ..
    } = agent;
    let biology = kind.biology();
    let mut senses = Senses {
        wind: climate.wind,
        up: world.up(position).to_scene(),
        aur: climate.aur,
        time: now,
        energy: body.energy,
        hydration: body.hydration,
        fatigue: body.fatigue,
        ..Senses::default()
    };
    for other in neighbours.iter().filter(|other| other.entity != entity) {
        let offset = (other.position - position).to_scene();
        if senses.food.is_none()
            && ((kind.eats(other.kind) && other.vitals.health > 0.0)
                || (kind.takes(Take::Carrion) && other.vitals.health <= 0.0))
        {
            senses.food = Some(offset);
            relations.remembered_food = Some((other.entity, world.position(other.position), now));
        }
        if senses.danger.is_none()
            && other.vitals.health > 0.0
            && other.kind.takes(Take::Prey)
            && other.kind.eats(kind)
        {
            senses.danger = Some(offset);
        }
        if senses.water.is_none()
            && plant(other.kind)
            && other.kind.organs().has(Function::Water)
            && other.vitals.health > 0.0
        {
            senses.water = Some(offset);
        }
        if senses.companion.is_none() && kind.partners(other.kind) && other.vitals.health > 0.0 {
            senses.companion = Some(offset);
            relations.companion = Some(other.entity);
            if offset.length_squared() < 25.0 && kind.attaches() {
                relations.host = Some(other.entity);
            }
        }
        if other.kind == kind && other.vitals.health > 0.0 {
            senses.separation -= offset / offset.length_squared().max(1.0);
            if kind.animal().is_some_and(|animal| animal.flocks) {
                senses.alignment += other.velocity.to_scene() * 0.02;
            }
        }
    }
    if senses.food.is_none()
        && let Some((remembered, location, seen)) = relations.remembered_food
        && now - seen < 8.0
    {
        if neighbours
            .iter()
            .any(|other| other.entity == remembered && other.vitals.health <= 0.0)
        {
            relations.remembered_food = None;
        } else {
            senses.food = Some((world.local(location) - position).to_scene());
        }
    }
    // An animal whose aur sacs run low skims the mists above the aur to fill them, high enough
    // that its dive cannot carry it into the aur itself.
    let target_altitude = if kind.animal().is_some_and(|animal| animal.aur > 0.0) && body.aur < 0.15
    {
        40.0
    } else {
        climate
            .altitude
            .clamp(biology.altitude.0 + 20.0, biology.altitude.1 - 20.0)
    };
    if (target_altitude - climate.altitude).abs() > 5.0 {
        senses.habitat = Some(senses.up * (target_altitude - climate.altitude));
    }
    if climate.storm > 0.1 {
        senses.storm = Some(Vec3::Y.to_scene() * 100.0);
    }
    let floats = kind.floats();
    senses.buoyancy = if kind.attaches() {
        0.0
    } else if floats {
        -climate.wind.dot(senses.up)
            + ((target_altitude - climate.altitude) * 0.01).clamp(-2.0, 2.0)
    } else if kind.animal().is_some_and(|animal| animal.flies()) {
        -climate.wind.dot(senses.up) - body.fatigue * 0.2
    } else {
        0.0
    };
    if kind.animal().is_some_and(|animal| animal.lets_go) && body.energy > 0.8 {
        relations.host = None;
    }
    senses
}

/// What an animal can perceive around it this step.
pub(super) struct Surroundings<'a> {
    pub(super) climate: Climate,
    pub(super) neighbours: &'a [Agent],
    pub(super) players: &'a [Vec3],
    pub(super) calls: &'a [Call],
    pub(super) now: f32,
}

/// What an organism makes of this step: its senses, how far it still depends on a parent, and, if
/// it has a mind of its own, its goal and the others it has fixed on.
pub(super) fn orient(
    agent: Agent,
    relations: &mut Relations,
    free: bool,
    around: &Surroundings,
    (index, world): (&HashMap<Entity, Agent>, &WorldData),
) -> (Senses, Stage, Targets) {
    let mut senses = perceive(
        agent,
        relations,
        around.climate,
        around.neighbours,
        around.now,
        world,
    );
    // The player and their mount go where the player steers, not where their bodies would take
    // them.
    if !free {
        senses.habitat = None;
    }
    let stage = nurture(agent, relations, index, around.now);
    let targets = if free && stage.forages() {
        think(agent, relations, &mut senses, around)
    } else {
        Targets::default()
    };
    stage.steer(&mut senses);
    relations.senses = senses;
    (senses, stage, targets)
}

/// Let an animal's mind weigh what it perceives against what its body asks, and set its goal;
/// returns the others it has fixed on.
pub(super) fn think(
    agent: Agent,
    relations: &mut Relations,
    senses: &mut Senses,
    around: &Surroundings,
) -> Targets {
    let kind = agent.kind;
    if plant(kind) {
        return Targets::default();
    }
    // Paralysed or tangled, it can do nothing: it hangs limp, and anything that cannot float
    // falls.
    if agent.vitals.stunned > 0.0 {
        senses.goal = Some(Goal::new(Activity::Rest, None, 0.0).folded(1.0));
        senses.power = 0.0;
        senses.sinking = agent.vitals.sinking(kind) + 6.0;
        return Targets::default();
    }
    if !relations.minded {
        relations.mind = Mind::new(agent.entity.to_bits() as u32);
        relations.minded = true;
    }
    let (seen, targets) = sense(agent, relations, senses, around);
    let drives = drives(agent, relations, around.now);
    let goal = relations.mind.decide(
        kind,
        drives,
        &seen,
        around.now,
        agent.entity.to_bits() as u32,
    );
    senses.goal = Some(goal);
    senses.power = agent.vitals.power(kind);
    senses.sinking = agent.vitals.sinking(kind);
    targets
}

/// The others an animal's mind has fixed on this step.
#[derive(Default)]
pub(super) struct Targets {
    pub(super) prey: Option<Agent>,
    pub(super) mate: Option<Agent>,
    pub(super) roost: Option<Entity>,
}

/// What an animal perceives this step, for its mind: prey, threats and how they move, mates, kin
/// and their mood, places to sleep, drink and guard, calls carried from far off and the aur's
/// phase. A sleeper notices only what comes close.
pub(super) fn sense(
    agent: Agent,
    relations: &Relations,
    senses: &Senses,
    around: &Surroundings,
) -> (Perception, Targets) {
    let Surroundings {
        climate,
        neighbours,
        ..
    } = *around;
    let kind = agent.kind;
    let offset = |other: Vec3| (other - agent.position).to_scene();
    let asleep = relations.activity == Activity::Sleep;
    let wary = if asleep {
        kind.senses() * 0.3
    } else {
        kind.senses()
    };
    let mut seen = first_sight(senses, climate);
    let mut targets = Targets::default();
    let ethology = kind.behaviour();
    let hunts = ethology.hunts;
    if kind.takes(Take::Prey) {
        seen.food = None;
    }
    let mut nearest = [f32::MAX; 6];
    let mut kin_sum = SceneVec3::ZERO;
    let mut heading = SceneVec3::ZERO;
    for other in neighbours
        .iter()
        .filter(|other| other.entity != agent.entity)
    {
        let to = offset(other.position);
        let distance = to.length();
        let sighting = Sighting {
            offset: to,
            velocity: other.velocity.to_scene(),
            helpless: other.vitals.stunned > 0.0,
        };
        let animal = !plant(other.kind);
        if !other.alive() {
            if animal && kind.eats(other.kind) && other.vitals.mass > 0.0 && distance < nearest[0] {
                nearest[0] = distance;
                seen.carcass = Some(to);
            }
            continue;
        }
        // Hunters take prey clearly smaller than themselves, and leave other hunters alone
        // unless starving.
        let fair = other.vitals.mass < agent.vitals.mass * 0.8
            && (!other.kind.takes(Take::Prey) || agent.vitals.energy < 0.2);
        let fair = fair || kind.animal().is_some_and(|animal| animal.parasite);
        if hunts
            && animal
            && fair
            && !other.protected
            && kind.eats(other.kind)
            && distance < nearest[1]
        {
            nearest[1] = distance;
            seen.prey = Some(sighting);
            targets.prey = Some(*other);
        }
        // A predator is a threat when it is hunting, or too close to trust.
        let menacing =
            matches!(other.activity, Activity::Hunt | Activity::Strike) || distance < 60.0;
        if distance < wary
            && distance < nearest[2]
            && menacing
            && other.kind.takes(Take::Prey)
            && other.kind.eats(kind)
        {
            nearest[2] = distance;
            seen.threat = Some(sighting);
        }
        if other.kind == kind {
            seen.kin_count += 1;
            kin_sum += to;
            heading += sighting.velocity;
            if other.activity == Activity::Flee {
                seen.fleeing += 1;
            }
            if other.grown() >= 1.0 && distance < nearest[3] {
                nearest[3] = distance;
                seen.mate = Some((to, other.activity == Activity::Display));
                targets.mate = Some(*other);
            }
        }
        (ethology.notice)(&glimpse(kind, other, to), &mut seen);
        if !animal && other.kind.biology().adult_mass >= 1.0 && distance < nearest[5] {
            nearest[5] = distance;
            targets.roost = Some(other.entity);
        }
    }
    if seen.kin_count > 0 {
        seen.kin = Some(kin_sum / seen.kin_count as f32);
        seen.heading = heading / seen.kin_count as f32;
    }
    sense_far(agent, relations, around, &mut seen);
    (seen, targets)
}

/// Another organism as it comes to an animal's notice, `to` away.
fn glimpse(kind: Kind, other: &Agent, to: SceneVec3) -> Glimpse {
    Glimpse {
        kind,
        other: other.kind,
        offset: to,
        grown: other.grown(),
        heard: false,
        activity: other.activity,
    }
}

/// What an animal knows before it looks about: the air, the aur's phase, and what its older senses
/// have found.
pub(super) fn first_sight(senses: &Senses, climate: Climate) -> Perception {
    Perception {
        up: senses.up,
        wind: senses.wind,
        food: senses.food,
        water: senses.water,
        habitat: senses.habitat,
        storm: senses.storm,
        light: climate.light,
        sunward: climate.sunward,
        updraft: climate.wind.dot(senses.up.normalize_or_zero()),
        humidity: climate.humidity,
        ..Perception::default()
    }
}

/// What an animal notices beyond its neighbours: its web, the player and the calls of its kind from
/// far off.
pub(super) fn sense_far(
    agent: Agent,
    relations: &Relations,
    around: &Surroundings,
    seen: &mut Perception,
) {
    let offset = |other: Vec3| (other - agent.position).to_scene();
    if let Some(home) = relations.home
        && let Some(web) = around.neighbours.iter().find(|other| other.entity == home)
    {
        seen.home = Some(offset(web.position));
    }
    let nearest = |a: &SceneVec3, b: &SceneVec3| a.length().total_cmp(&b.length());
    seen.player = around
        .players
        .iter()
        .map(|&player| offset(player))
        .filter(|to| to.length() < agent.kind.senses() * 1.5)
        .min_by(nearest);
    // Calls carry far, and each lineage makes what it will of the ones it hears.
    let notice = agent.kind.behaviour().notice;
    for call in around.calls {
        let to = offset(call.at);
        if call.at.distance(agent.position) > 20.0 && to.length() < call.reach {
            let heard = Glimpse {
                kind: agent.kind,
                other: call.caller,
                offset: to,
                grown: 1.0,
                heard: true,
                activity: Activity::Display,
            };
            notice(&heard, seen);
        }
    }
}

/// What the body asks of the mind: food, water, rest and sleep, relief from its wounds, and, in a
/// grown, well-fed animal not already carrying young, a mate.
pub(super) fn drives(agent: Agent, relations: &Relations, now: f32) -> Drives {
    let body = agent.vitals;
    let viscera = agent.kind.organs();
    let damage = viscera
        .sites()
        .iter()
        .enumerate()
        .map(|(index, _)| 1.0 - body.organs[index])
        .sum::<f32>()
        / viscera.sites().len().max(1) as f32;
    let youth = agent.grown().min(1.0);
    let ready = youth >= 1.0
        && relations.fertile_until < now
        && now - relations.last_birth > 90.0
        && body.working(&viscera, Function::Breeding) > 0.4;
    Drives {
        hunger: (1.0 - body.energy)
            * if body.gut > body.mass * 0.05 {
                0.6
            } else {
                1.0
            },
        thirst: 1.0 - body.hydration,
        tiredness: body.fatigue,
        sleepiness: 1.0 - body.alertness,
        hurt: ((1.0 - body.fluid) * 1.5 + body.bleeding * 20.0 + damage).min(1.5),
        desire: if ready {
            ((body.energy - 0.55) * 2.5).clamp(0.0, 1.0) * body.hydration.min(1.0)
        } else {
            0.0
        },
        youth,
    }
}
