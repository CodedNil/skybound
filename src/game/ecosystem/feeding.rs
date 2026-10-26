//! Eating: what each feeder takes from a plant or a carcass, and the meals settled each step.

use crate::game::ecosystem::{Agent, Organisms, Pool, Relations};
use bevy::prelude::*;
use skybound::scene::life::{
    Kind,
    animalia::physiology::{Take, can_eat, plant},
    behaviour::Activity,
    organs::{Function, Viscera},
    vitals::Vitals,
};

pub(super) struct Meal {
    pub(super) eater: Entity,
    pub(super) food: Entity,
    pub(super) amount: f32,
}

/// Whether an animal can eat another as it is now: a predator only once it has been killed or
/// made helpless, a scavenger only dead, anything else only alive.
pub(super) fn edible_now(kind: Kind, other: &Agent) -> bool {
    can_eat(kind, other.kind, other.alive(), other.vitals.stunned > 0.0)
}

/// What a feeder takes from a plant, and the damage it does the organ it eats: a grazer crops
/// fronds, a bark-stripper the stem's bark, a Grounder the Root-Star's roots, a nectar-feeder
/// drinks nectar, and fruit-eaters take the fruit. Drinking at a Pondplant draws down its cistern.
/// Returns how much was eaten.
pub(super) fn grazed(plant: &mut Vitals, kind: Kind, eater: Kind, amount: f32) -> f32 {
    let viscera = kind.organs();
    if amount <= 0.0 {
        drain(plant, &viscera, Function::Water, 0.003);
        return 0.0;
    }
    if eater.pollinates() {
        let taken = amount.min(plant.reserve * plant.mass * 0.05);
        plant.reserve = (plant.reserve - taken / (plant.mass * 0.05).max(1e-6)).max(0.0);
        return taken;
    }
    let crops = eater
        .animal()
        .map_or(Function::Gathering, |animal| animal.crops);
    let ripe = viscera.has(Function::Fruit) && plant.working(&viscera, Function::Fruit) > 0.2;
    let organ = if crops == Function::Gathering && ripe {
        Function::Fruit
    } else {
        crops
    };
    let left = plant.working(&viscera, organ);
    let taken = amount * left;
    drain(plant, &viscera, organ, taken / plant.mass.max(1e-6) * 4.0);
    taken
}

/// Take a share away from every organ meeting one need.
pub(super) fn drain(body: &mut Vitals, viscera: &Viscera, function: Function, share: f32) {
    for (index, site) in viscera.sites().iter().enumerate() {
        if site.organ.function == function {
            body.organs[index] = (body.organs[index] - share).max(0.0);
        }
    }
}

pub(super) fn forage(
    agent: Agent,
    body: &mut Vitals,
    relations: &Relations,
    neighbours: &[Agent],
    pool: &mut Pool,
    dt: f32,
) -> Option<Meal> {
    let Agent {
        entity,
        kind,
        position,
        ..
    } = agent;
    let senses = &relations.senses;
    let biology = kind.biology();
    if relations.activity == Activity::Drink
        && senses
            .water
            .is_some_and(|offset| offset.length_squared() < 100.0)
        && let Some(pond) = neighbours.iter().find(|other| {
            other.kind.organs().has(Function::Water)
                && plant(other.kind)
                && other.position.distance_squared(position) < 225.0
        })
    {
        // Only as much as the Pondplant's cistern still holds.
        let water = pond.vitals.working(&pond.kind.organs(), Function::Water);
        body.hydration = (body.hydration + dt * 0.2 * water).min(1.0);
        return Some(Meal {
            eater: entity,
            food: pond.entity,
            amount: 0.0,
        });
    }
    if kind.takes(Take::Carrion) && body.energy < 0.85 {
        let amount = pool.detritus.min(body.mass * dt * 0.003);
        let aur = pool.bound_aur * amount / pool.detritus.max(1e-10);
        pool.detritus -= amount;
        pool.bound_aur -= aur;
        body.ingest(amount, aur);
    }
    if kind.plant().is_some_and(|plant| plant.feeds_host)
        && let Some(host) = relations.host
    {
        return Some(Meal {
            eater: host,
            food: entity,
            amount: body.mass * dt * 0.0001,
        });
    }
    if !matches!(
        relations.activity,
        Activity::Feed | Activity::Clean | Activity::Attach | Activity::Drift
    ) {
        return None;
    }
    let reach = biology.adult_mass.cbrt() * 2.0 + 2.0;
    neighbours
        .iter()
        .find(|other| {
            other.entity != entity
                && !other.protected
                && edible_now(kind, other)
                && other.position.distance_squared(position) < reach * reach
        })
        .map(|food| Meal {
            eater: entity,
            food: food.entity,
            amount: body.mass * dt * 0.003,
        })
}

pub(super) fn transfer_meals(organisms: &mut Query<Organisms>, meals: &[Meal], now: f32) -> u64 {
    let mut deaths = 0;
    for meal in meals {
        let Ok(
            [
                (_, eater_kind, mut eater, mut visitor, ..),
                (_, food_kind, mut food, mut flower, ..),
            ],
        ) = organisms.get_many_mut([meal.eater, meal.food])
        else {
            continue;
        };
        if eater.0.health <= 0.0 || food.0.mass <= 0.0 {
            continue;
        }
        let nectar = eater_kind.0.pollinates();
        if !nectar && food.0.health <= 0.0 && eater_kind.0.takes(Take::Living) {
            continue;
        }
        let rate = if nectar { 0.1 } else { 1.0 };
        let mut amount = (meal.amount * rate).min(food.0.mass * 0.05);
        if plant(food_kind.0) {
            amount = grazed(&mut food.0, food_kind.0, eater_kind.0, amount);
            if food_kind.0.organs().has(Function::Fruit) && !nectar {
                // The seed rides in the gut and is passed somewhere else.
                visitor.seed = Some((food_kind.0, now + 150.0));
            }
        }
        food.0.mass -= amount;
        eater.0.ingest(
            amount,
            if plant(food_kind.0) {
                amount * 0.1
            } else {
                0.0
            },
        );
        if nectar {
            if let Some((donor, kind)) = visitor.pollen
                && donor != meal.food
                && kind == food_kind.0
            {
                flower.pollinated_until = now + 120.0;
            }
            visitor.pollen = Some((meal.food, food_kind.0));
        }
        if eater_kind.0.animal().is_some_and(|animal| animal.parasite) {
            visitor.host = Some(meal.food);
        }
        if food.0.mass < food_kind.0.biology().adult_mass * 0.25 && food.0.health > 0.0 {
            food.0.health = 0.0;
            deaths += 1;
        }
    }
    deaths
}
