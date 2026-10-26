//! Moving organisms for a step: under their own power, carried, or adrift.

use crate::{
    game::{
        ecosystem::{Agent, EcologicalMotion, Relations, breeding::Stage},
        world::{PlanetPosition, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::prelude::*;
use core::f32::consts::TAU;
use skybound::scene::life::{
    Kind,
    animalia::physiology::{Brood, Take, plant},
    behaviour::Motion,
    genome::{BUILD, HUE, SHADE},
};
use spirv_std::glam::Vec3 as SceneVec3;
use std::collections::HashMap;

/// Everything moving an organism needs from the world.
pub(super) struct Move<'a> {
    /// Its place on the planet, once it has one.
    pub(super) coordinate: Option<&'a mut PlanetPosition>,
    pub(super) world: &'a WorldData,
    pub(super) wind: Vec3,
    pub(super) dt: f32,
    pub(super) now: f32,
}

/// Move an organism for this step: carried by whatever holds it, adrift as an egg, or under its
/// own power.
pub(super) fn relocate(
    agent: Agent,
    stage: Stage,
    relations: &mut Relations,
    (transform, motion): (&mut Transform, &mut EcologicalMotion),
    movement: &Motion,
    index: &HashMap<Entity, Agent>,
    step: Move,
) {
    let Move {
        coordinate,
        world,
        wind,
        dt,
        now,
    } = step;
    let (was, turned) = (transform.translation, transform.rotation);
    if let Some(host) = relations
        .host
        .and_then(|host| index.get(&host))
        .filter(|host| host.alive())
    {
        let carried = relations.parent == relations.host;
        transform.translation = held_position(agent, host, carried, world);
        if carried
            && agent
                .kind
                .animal()
                .is_some_and(|animal| matches!(animal.brood, Brood::Strings | Brood::Pouch))
        {
            transform.rotation = host.rotation;
        }
        motion.0 = host.velocity;
    } else if stage == Stage::Egg {
        // An egg cast loose goes where the air takes it.
        travel(wind * dt, transform, coordinate, world);
        motion.0 = wind;
    } else {
        relations.host = None;
        let offset = movement.position.to_engine();
        travel(offset, transform, coordinate, world);
        transform.rotation = movement.rotation.to_engine();
        motion.0 = movement.velocity.to_engine();
    }
    // Remembered so it is drawn gliding from where it was to here until the next step.
    let lag = was - transform.translation;
    relations.lag = if lag.length() < 200.0 {
        lag
    } else {
        Vec3::ZERO
    };
    relations.turned = turned;
    relations.stepped = now;
}

/// What the step's motion does to the body itself: an animal's wings beat on through their
/// stroke, faster and deeper the harder it is working them and barely at all as it glides; a
/// plant bends against the air moving past it, springing back and overshooting as it eases.
pub(super) fn flex(
    relations: &mut Relations,
    kind: Kind,
    movement: &Motion,
    wind: SceneVec3,
    dt: f32,
) {
    relations.effort += (movement.effort - relations.effort) * (dt * 2.0).min(1.0);
    if plant(kind) {
        // The air pushing past it, in its own frame, sets where the spring is pulled to.
        let air = movement.rotation.inverse() * (wind - movement.velocity);
        let reach = kind.render_radius() * 0.12;
        let pull = (air.to_engine() * 0.25).clamp_length_max(reach);
        let stiffness = 3.0;
        let damping = 1.2;
        relations.bending +=
            ((pull - relations.bend) * stiffness - relations.bending * damping) * dt;
        relations.bend += relations.bending * dt;
    } else {
        let rate = kind.look().flight.x * (0.6 + relations.effort.min(1.0) * 0.8);
        // Kept over two beats, so wings beating half as fast again to make up for a weak side
        // come round in step.
        relations.wingbeat = (relations.wingbeat + rate * TAU * dt) % (2.0 * TAU);
    }
}

/// An organism's motion this step in the scene's frame, starting from where it is, at its velocity
/// and heading.
pub(super) fn moving(velocity: Vec3, rotation: Quat) -> Motion {
    Motion {
        position: Vec3::ZERO.to_scene(),
        velocity: velocity.to_scene(),
        rotation: rotation.to_scene(),
        effort: 0.0,
    }
}

/// Where an organism held by another sits. Carried by its parent, it is in its pouch or glued to
/// its body, or on a string trailing behind it; clinging to a plant or animal, it is somewhere on
/// it, each at its own place.
pub(super) fn held_position(agent: Agent, host: &Agent, carried: bool, world: &WorldData) -> Vec3 {
    if carried {
        return match agent.kind.animal().map(|animal| animal.brood) {
            Some(Brood::Strings) => {
                host.position + host.rotation * Vec3::Z * host.kind.render_radius() * 0.5
            }
            // In the pouch, its front at the pouch's mouth.
            Some(Brood::Pouch) => {
                let pouch = host.kind.look().brood * host.kind.biology().size;
                host.position
                    + host.rotation
                        * Vec3::new(0.0, pouch.x + pouch.w * 0.3, pouch.y - pouch.z * 0.4)
            }
            _ => host.position,
        };
    }
    // Prey is held to the hunter's jaws by its mouth tentacles, beneath and ahead of its head.
    if host.kind.takes(Take::Prey) {
        let reach = host.kind.render_radius();
        return host.position + host.rotation * Vec3::new(0.0, -reach * 0.15, -reach * 0.45);
    }
    // Anything else rides on top, each where its own build lets it cling best: further out or in,
    // fore or aft.
    let genes = agent.genes;
    let around = Vec3::new(
        (genes.get(HUE) - 0.5) * 6.0,
        (genes.get(SHADE) - 0.5) * 4.0,
        (genes.get(BUILD) - 0.5) * 2.0,
    ) * 0.3;
    let up = world.up(host.position);
    host.position + (up + around) * host.kind.biology().adult_mass.cbrt() * 0.3
}

/// Move an organism through the world, on the planet's surface frame when it has one.
pub(super) fn travel(
    offset: Vec3,
    transform: &mut Transform,
    coordinate: Option<&mut PlanetPosition>,
    world: &WorldData,
) {
    if let Some(coordinate) = coordinate {
        coordinate.translate(offset, transform, world);
    } else {
        transform.translation += offset;
    }
}
