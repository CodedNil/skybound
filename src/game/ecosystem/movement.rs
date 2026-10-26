//! Moving organisms for a step: under their own power, carried, or adrift.

use crate::{
    game::{
        ecosystem::{Agent, EcologicalMotion, Relations, breeding::Stage},
        world::{PlanetPosition, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::prelude::*;
use skybound::scene::life::{
    animalia::physiology::{Brood, Take},
    behaviour::Motion,
};
use std::collections::HashMap;

/// Everything moving an organism needs from the world.
pub(super) struct Move<'a, 'w, 's, 'q> {
    pub(super) coordinates: &'a mut Query<'w, 's, &'q mut PlanetPosition>,
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
        coordinates,
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
        travel(agent.entity, wind * dt, transform, coordinates, world);
        motion.0 = wind;
    } else {
        relations.host = None;
        let offset = movement.position.to_engine();
        travel(agent.entity, offset, transform, coordinates, world);
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

/// An organism's motion this step in the scene's frame, starting from where it is, at its velocity
/// and heading.
pub(super) fn moving(velocity: Vec3, rotation: Quat) -> Motion {
    Motion {
        position: Vec3::ZERO.to_scene(),
        velocity: velocity.to_scene(),
        rotation: rotation.to_scene(),
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
                let pouch = host.kind.look().brood;
                host.position
                    + host.rotation
                        * Vec3::new(0.0, pouch.x + pouch.w * 0.3, pouch.y - pouch.z * 0.4)
            }
            _ => host.position,
        };
    }
    let bits = agent.entity.to_bits();
    let around = Vec3::new(
        (bits % 7) as f32 - 3.0,
        (bits % 5) as f32 - 2.0,
        (bits % 3) as f32 - 1.0,
    ) * 0.3;
    // Prey hangs beneath the hunter that holds it; anything else rides on top.
    let up = if host.kind.takes(Take::Prey) {
        -world.up(host.position)
    } else {
        world.up(host.position)
    };
    host.position + (up + around) * host.kind.biology().adult_mass.cbrt() * 0.3
}

/// Move an organism through the world, on the planet's surface frame when it has one.
pub(super) fn travel(
    entity: Entity,
    offset: Vec3,
    transform: &mut Transform,
    coordinates: &mut Query<&mut PlanetPosition>,
    world: &WorldData,
) {
    if let Ok(mut coordinate) = coordinates.get_mut(entity) {
        coordinate.translate(offset, transform, world);
    } else {
        transform.translation += offset;
    }
}
