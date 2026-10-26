//! Breeding and the young: where newborns are kept, how they grow independent, and births.

use crate::{
    game::ecosystem::{Agent, Pool, Relations, feeding::Meal, population},
    math::ToScene,
};
use bevy::prelude::*;
use skybound::scene::life::{
    animalia::{
        BOLDNESS, CURIOSITY,
        physiology::{Brood, plant},
    },
    behaviour::Senses,
    biology::Biology,
    genome::Genome,
    organs::Function,
    plantae::physiology::Propagation,
    vitals::Vitals,
};
use std::collections::HashMap;

/// Bring a young one into the world beside its parent, with genes from its parent and from the
/// mate or pollen that fathered it; a plant dividing or breaking apart passes on its own, changed a
/// little.
pub(super) fn bear(
    commands: &mut Commands,
    parent: Agent,
    sire: Option<Genome>,
    neighbours: &[Agent],
    position: Vec3,
    now: f32,
) {
    let seed = parent.entity.to_bits() ^ u64::from(now.to_bits()).rotate_left(32);
    let genes = Genome::child(parent.genes, sire.unwrap_or(parent.genes), seed);
    commands
        .spawn(population(
            parent.kind,
            position + Vec3::X * 3.0,
            0.15,
            genes,
        ))
        .insert(Relations {
            host: nest(parent, neighbours),
            parent: Some(parent.entity),
            genes,
            ..Relations::default()
        });
}

/// Where a newborn is first kept, by its lineage's way of keeping them: on its parent, tethered to
/// the nearest living plant (its food plant if there is one in reach), or nowhere.
pub(super) fn nest(parent: Agent, neighbours: &[Agent]) -> Option<Entity> {
    match parent.kind.animal()?.brood {
        Brood::Carried | Brood::Strings | Brood::Pouch => Some(parent.entity),
        Brood::Tethered => {
            let rank = |other: &Agent| {
                let preference = if parent.kind.eats(other.kind) {
                    0.2
                } else {
                    1.0
                };
                other.position.distance(parent.position) * preference
            };
            neighbours
                .iter()
                .filter(|other| plant(other.kind) && other.vitals.health > 0.0)
                .min_by(|a, b| rank(a).total_cmp(&rank(b)))
                .map(|plant| plant.entity)
        }
        Brood::Live | Brood::Drifting => None,
    }
}

/// How far a young organism still depends on its parent.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Stage {
    /// Grown, or never cared for.
    Free,
    /// An egg, carried or cast loose, living on its yolk.
    Egg,
    /// In its parent's pouch, fed there.
    Pouch(Entity),
    /// Out of the pouch and due back: the way to the parent.
    Homing(Vec3),
}

impl Stage {
    /// Back to the pouch, whatever else it might rather be doing.
    pub(super) fn steer(self, senses: &mut Senses) {
        if let Self::Homing(offset) = self {
            senses.habitat = Some(offset.to_scene());
            senses.danger = None;
            senses.separation *= 0.0;
        }
    }

    pub(super) const fn forages(self) -> bool {
        !matches!(self, Self::Egg | Self::Pouch(_))
    }

    /// What a young organism that does not forage lives on: an egg on its yolk, and pouch young on
    /// what the parent feeds them.
    pub(super) fn provide(self, entity: Entity, body: &mut Vitals, dt: f32) -> Option<Meal> {
        if let Self::Pouch(parent) = self {
            return Some(Meal {
                eater: entity,
                food: parent,
                amount: body.mass * dt * 0.002,
            });
        }
        body.energy = body.energy.max(0.6);
        body.hydration = body.hydration.max(0.6);
        None
    }
}

/// Where a young organism is kept, by its lineage's way of keeping them. Eggs stay where they were
/// laid until they hatch, except strings, which the parent cuts loose to drift soon after laying
/// them. Pouch young stay in the pouch until they are formed, then make trips out to feed, at first
/// going out while fed and homing on the parent as they grow hungry, until they no longer come
/// back.
pub(super) fn nurture(
    agent: Agent,
    relations: &mut Relations,
    index: &HashMap<Entity, Agent>,
) -> Stage {
    let biology = agent.kind.biology();
    let young = (agent.vitals.age_days / f64::from(biology.maturity_days)) as f32;
    let brood = agent.kind.animal().map(|animal| animal.brood);
    let Some(parent) = relations
        .parent
        .and_then(|parent| index.get(&parent))
        .filter(|parent| parent.vitals.health > 0.0)
    else {
        relations.parent = None;
        let hatches = agent.kind.animal().is_some_and(|animal| animal.hatches);
        let egg = hatches && brood.is_some_and(Brood::eggs) && young < 0.3;
        // Orphaned or hatched, a young tetralate flies free of whatever it was laid on.
        if hatches && !egg {
            relations.host = None;
        }
        return if egg { Stage::Egg } else { Stage::Free };
    };
    let parent = *parent;
    match brood {
        Some(Brood::Pouch) if young < 0.2 => {
            relations.host = Some(parent.entity);
            Stage::Pouch(parent.entity)
        }
        Some(Brood::Pouch) if young < 0.7 => {
            // It ventures out while it is fed and bold enough for how grown it is, and comes back
            // to the pouch when it grows hungry: the bolder and more curious, the longer it stays
            // out.
            let independence = (young - 0.2) / 0.5;
            let daring = agent.genes.get(BOLDNESS) * 0.5 + agent.genes.get(CURIOSITY) * 0.5;
            let leaves = 1.0 - independence * (0.5 + daring * 0.5);
            if agent.vitals.energy > leaves {
                relations.host = None;
                return Stage::Free;
            }
            let offset = parent.position - agent.position;
            if relations.host == Some(parent.entity)
                || offset.length() < parent.kind.render_radius() * 0.4 + 3.0
            {
                relations.host = Some(parent.entity);
                Stage::Pouch(parent.entity)
            } else {
                Stage::Homing(offset)
            }
        }
        Some(Brood::Pouch | Brood::Live) => {
            relations.parent = None;
            relations.host = None;
            Stage::Free
        }
        _ if young < 0.3 => {
            if brood == Some(Brood::Strings) && young > 0.08 {
                relations.host = None;
            }
            Stage::Egg
        }
        _ => {
            relations.parent = None;
            relations.host = None;
            Stage::Free
        }
    }
}

/// A body decaying into the patch around it; returns whether nothing is left of it.
pub(super) fn decompose(body: &mut Vitals, pool: &mut Pool, dt: f32, biology: Biology) -> bool {
    let amount = body.mass * (1.0 - (-dt * 0.005).exp());
    body.mass -= amount;
    pool.detritus += amount + body.gut;
    pool.bound_aur += body.gut_aur;
    body.gut = 0.0;
    body.gut_aur = 0.0;
    body.mass < biology.adult_mass * 0.001
}

pub(super) fn reproduce(
    agent: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    neighbours: &[Agent],
    now: f32,
) -> bool {
    let kind = agent.kind;
    let biology = kind.biology();
    let maturity = f64::from(biology.maturity_days);
    let mature = body.age_days >= maturity && body.mass > biology.adult_mass * 1.15;
    let fertile = match (kind.animal(), kind.plant()) {
        // Only after courting a mate to the end.
        (Some(animal), _) => !animal.mates || relations.fertile_until > now,
        (_, Some(plant)) => {
            plant.propagation != Propagation::Pollination || relations.pollinated_until > now
        }
        _ => true,
    };
    let brood = agent.kind.organs();
    let whole = body.working(&brood, Function::Breeding) > 0.4;
    if mature
        && fertile
        && whole
        && body.energy > 0.8
        && body.hydration > 0.5
        && now - relations.last_birth > 90.0
    {
        body.mass -= biology.adult_mass * 0.15;
        body.energy -= 0.2;
        relations.last_birth = now;
        let _ = neighbours;
        return true;
    }
    false
}
