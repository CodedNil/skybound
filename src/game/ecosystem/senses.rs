//! What an organism senses and wants: its neighbours, the air, its body's drives, and what its
//! mind makes of them.

use crate::{
    game::{
        ecosystem::{
            Agent, Call, Index, Relations,
            breeding::{Stage, nurture},
        },
        world::WorldData,
    },
    math::{ToEngine, ToScene},
};
use bevy::{math::DVec3, prelude::*};
use skybound::scene::{
    PLANET_RADIUS,
    climate::Climate,
    features::{nearest_geyser, nearest_lake, nearest_vent_field},
    life::{
        Kind,
        animalia::{
            BOLDNESS,
            mind::{Drives, Glimpse, Perception, Sighting},
            physiology::{Take, plant},
        },
        behaviour::{Activity, Goal, Senses},
        organs::Function,
    },
};
use spirv_std::glam::Vec3 as SceneVec3;

/// How hard an animal steers away from another body `offset` from it, so bodies keep out of one
/// another, the larger needing the more room; plants hold their places.
fn keep_clear(kind: Kind, other: &Agent, offset: SceneVec3) -> SceneVec3 {
    let room = (kind.render_radius() + other.kind.render_radius()) * 0.6;
    let distance = offset.length();
    if distance < room && other.vitals.health > 0.0 && !plant(kind) {
        -offset / distance.max(0.01) * (1.0 - distance / room) * 8.0
    } else {
        SceneVec3::ZERO
    }
}

pub(super) fn perceive(
    agent: Agent,
    relations: &mut Relations,
    climate: Climate,
    neighbours: &[&Agent],
    now: f32,
    world: &WorldData,
) -> Senses {
    let Agent {
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
        gas: body.lift,
        wings: body.wings().0,
        list: body.wings().1,
        ..Senses::default()
    };
    survey(agent, relations, neighbours, &mut senses, (now, world));
    sense_features(kind, position, world, &mut senses);
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
    senses.storm = storm(climate, senses.up);
    senses.buoyancy = climb(
        kind,
        climate,
        senses.up,
        target_altitude - climate.altitude,
        body.fatigue,
    );
    if kind.animal().is_some_and(|animal| animal.lets_go) && body.energy > 0.8 {
        relations.host = None;
    }
    senses
}

/// Look over the neighbours: the nearest food, danger, water and companion, for neighbours come in
/// no particular order, and how hard to steer to keep clear of the rest and keep with its own kind.
fn survey(
    agent: Agent,
    relations: &mut Relations,
    neighbours: &[&Agent],
    senses: &mut Senses,
    (now, world): (f32, &WorldData),
) {
    let Agent {
        entity,
        kind,
        position,
        ..
    } = agent;
    let mut nearest = [f32::MAX; 4];
    let mut food: Option<&Agent> = None;
    let mut companion: Option<&Agent> = None;
    for other in neighbours.iter().filter(|other| other.entity != entity) {
        let offset = (other.position - position).to_scene();
        let distance = offset.length_squared();
        let alive = other.vitals.health > 0.0;
        if distance < nearest[0]
            && ((kind.eats(other.kind) && alive) || (kind.takes(Take::Carrion) && !alive))
        {
            nearest[0] = distance;
            senses.food = Some(offset);
            food = Some(other);
        }
        if distance < nearest[1] && alive && other.kind.takes(Take::Prey) && other.kind.eats(kind) {
            nearest[1] = distance;
            senses.danger = Some(offset);
        }
        if distance < nearest[2]
            && alive
            && plant(other.kind)
            && other.kind.organs().has(Function::Water)
        {
            nearest[2] = distance;
            senses.water = Some(offset);
        }
        if distance < nearest[3] && alive && kind.partners(other.kind) {
            nearest[3] = distance;
            senses.companion = Some(offset);
            companion = Some(other);
        }
        senses.separation += keep_clear(kind, other, offset);
        if other.kind == kind && alive {
            senses.separation -= offset / distance.max(1.0);
            if kind.animal().is_some_and(|animal| animal.flocks) {
                senses.alignment += other.velocity.to_scene() * 0.02;
            }
        }
    }
    if let Some(food) = food {
        relations.remembered_food = Some((food.entity, world.position(food.position), now));
    }
    if let Some(companion) = companion {
        relations.companion = Some(companion.entity);
        if nearest[3] < 25.0 && kind.attaches() {
            relations.host = Some(companion.entity);
        }
    }
}

/// Which way a storm is: its heart, to get away from; without one to place, up and out of the
/// rain.
fn storm(climate: Climate, up: SceneVec3) -> Option<SceneVec3> {
    (climate.storm > 0.1).then(|| {
        if climate.storm_toward.length_squared() > 0.5 {
            climate.storm_toward * 400.0
        } else {
            up * 100.0
        }
    })
}

/// What the planet's features mean to a body: a geyser's jet, which kills whatever touches it, is
/// given a wide berth by everything that can steer, and a floating lake is water to drink where no
/// basin is near.
fn sense_features(kind: Kind, position: Vec3, world: &WorldData, senses: &mut Senses) {
    let planet = world.position(position);
    let up = planet.normalize().as_vec3().to_scene();
    let altitude = (planet.length() - f64::from(PLANET_RADIUS)) as f32;
    let toward = |target: DVec3| {
        world
            .rotate_to_local((target - planet).as_vec3())
            .to_scene()
    };
    if let Some((geyser, distance)) = nearest_geyser(up) {
        let room = geyser.radius + kind.render_radius() + 40.0;
        if distance < room && altitude < geyser.height + 50.0 {
            let axis = geyser.up.to_engine().as_dvec3() * planet.length();
            let away = -toward(axis);
            senses.separation += away / away.length().max(1e-3) * (1.0 - distance / room) * 30.0;
        }
    }
    // The nearest Mineral Field, which the dust in the air points the way to from far off.
    let (field, distance) = nearest_vent_field(up);
    if distance < 60_000.0 {
        let heart = field.up.to_engine().as_dvec3() * planet.length();
        senses.minerals = Some(toward(heart));
    }
    let (lake, distance) = nearest_lake(up);
    if senses.water.is_none()
        && distance < lake.radius + 400.0
        && (altitude - lake.altitude).abs() < 600.0
    {
        // The nearest of its shore, on the water.
        let centre = lake.up.to_engine().as_dvec3();
        let out = (up.to_engine().as_dvec3() - centre).normalize_or_zero();
        let reach = f64::from(distance.min(lake.radius * 0.8) / PLANET_RADIUS);
        let shore =
            (centre + out * reach).normalize() * f64::from(PLANET_RADIUS + lake.altitude + 1.0);
        senses.water = Some(toward(shore));
    }
}

/// How fast it means to rise or sink against the air's own rising: a floater trims its gas
/// towards the height it lives at, a flier holds its height against the updraft, sagging as it
/// tires, and anything fastened to another goes where its host goes.
fn climb(kind: Kind, climate: Climate, up: SceneVec3, below: f32, fatigue: f32) -> f32 {
    if kind.attaches() {
        0.0
    } else if kind.floats() {
        -climate.wind.dot(up) + (below * 0.01).clamp(-2.0, 2.0)
    } else if kind.animal().is_some_and(|animal| animal.flies()) {
        -climate.wind.dot(up) - fatigue * 0.2
    } else {
        0.0
    }
}

/// What an animal can perceive around it this step.
pub(super) struct Surroundings<'a> {
    pub(super) climate: Climate,
    pub(super) neighbours: &'a [&'a Agent],
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
    (index, world): (&Index, &WorldData),
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
    let stage = nurture(agent, relations, index);
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
        return Targets::default();
    }
    let (seen, targets) = sense(agent, relations, senses, around);
    let drives = drives(agent, relations, around.now);
    let goal = relations
        .mind
        .decide(kind, drives, &seen, around.now, agent.genes);
    senses.goal = Some(goal);
    senses.power = agent.vitals.power(kind);
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
    let mut nearest = [f32::MAX; 7];
    let mut kin_sum = SceneVec3::ZERO;
    let mut heading = SceneVec3::ZERO;
    let respects = Sanctuaries::around(neighbours, agent);
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
        let fair = other.vitals.mass < agent.vitals.mass * kind.behaviour().largest
            && (!other.kind.takes(Take::Prey) || agent.vitals.energy < 0.2);
        let fair = fair || kind.animal().is_some_and(|animal| animal.parasite);
        if hunts
            && animal
            && fair
            && !other.protected
            && respects.allow(other.position)
            && kind.eats(other.kind)
            && distance < nearest[1]
        {
            nearest[1] = distance;
            seen.prey = Some(sighting);
            targets.prey = Some(**other);
        }
        if distance < wary && distance < nearest[2] && menaces(other, kind, distance) {
            nearest[2] = distance;
            seen.threat = Some(sighting);
        }
        let shelters = other.kind.behaviour().shelters;
        if shelters > 0.0
            && other.grown() >= 1.0
            && distance < shelters * 1.5
            && distance < nearest[6]
        {
            nearest[6] = distance;
            seen.sanctuary = Some(to);
        }
        if other.kind == kind {
            seen.kin_count += 1;
            kin_sum += to;
            heading += sighting.velocity;
            if other.activity == Activity::Flee {
                seen.fleeing += 1;
            }
            // A mate is one grown and of a sex it can breed with, as its lineage's sexes have it.
            if other.grown() >= 1.0
                && distance < nearest[3]
                && agent.vitals.suits(&other.vitals, kind)
            {
                nearest[3] = distance;
                seen.mate = Some((to, other.activity == Activity::Display));
                targets.mate = Some(**other);
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

/// The guardians about a hunter, each keeping the young of every kind safe for some way around it,
/// and whether the hunter is hungry enough, and bold enough, to test them as most will not.
struct Sanctuaries {
    guardians: Vec<(Vec3, f32)>,
    desperate: bool,
}

impl Sanctuaries {
    fn around(neighbours: &[&Agent], hunter: Agent) -> Self {
        Self {
            guardians: neighbours
                .iter()
                .filter(|other| other.alive() && other.grown() >= 1.0)
                .map(|other| (other.position, other.kind.behaviour().shelters))
                .filter(|&(_, shelters)| shelters > 0.0)
                .collect(),
            desperate: hunter.vitals.energy < 0.08 + hunter.genes.get(BOLDNESS) * 0.17,
        }
    }

    /// Whether the hunter will hunt at a place: anywhere outside a sanctuary, and in one only if
    /// desperate.
    fn allow(&self, at: Vec3) -> bool {
        self.desperate
            || !self
                .guardians
                .iter()
                .any(|&(guardian, shelters)| guardian.distance(at) < shelters)
    }
}

/// Whether another organism `distance` away menaces an animal: a predator that would eat it, when
/// hunting or too close to trust, or, to a hunter inside a sanctuary, a guardian standing off a
/// hunt. A guardian only patrolling troubles no one.
fn menaces(other: &Agent, kind: Kind, distance: f32) -> bool {
    let hunting = matches!(other.activity, Activity::Hunt | Activity::Strike);
    let predator = other.kind.takes(Take::Prey) && other.kind.eats(kind);
    let driving_off = kind.behaviour().hunts
        && other.kind != kind
        && distance < other.kind.behaviour().shelters
        && other.activity == Activity::Defend;
    (predator && (hunting || distance < 60.0)) || driving_off
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
        minerals: senses.minerals,
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
        && body.working(viscera, Function::Breeding) > 0.4;
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
        minerals: if agent.kind.animal().is_some_and(|animal| animal.ironclad) {
            (1.0 - body.iron * 1.6).clamp(0.0, 1.0)
        } else {
            0.0
        },
        term: if body.expecting > 0 {
            body.gestation
        } else {
            0.0
        },
        // Remembered for an hour or so, as long as its young are small.
        borne: if relations.last_birth > 0.0 {
            (1.0 - (now - relations.last_birth) / 3600.0).clamp(0.0, 1.0)
        } else {
            0.0
        },
    }
}
