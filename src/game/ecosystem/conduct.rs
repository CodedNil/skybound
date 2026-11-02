//! What an animal does once it has decided: strike and hold its prey, court, call and sleep.

use crate::{
    game::ecosystem::{Agent, Attack, Call, Organisms, Relations, senses::Targets},
    math::{ToEngine, ToScene},
};
use bevy::prelude::*;
use skybound::scene::life::{
    Injury, Kind,
    animalia::{
        AGILITY, PRECISION,
        mind::{Aim, Snare},
        physiology::{Take, plant},
    },
    behaviour::Activity,
    organs::{Function, Strike},
    vitals::Vitals,
};
use spirv_std::glam::Vec3 as SceneVec3;
use std::collections::HashMap;

/// A call carries as far as the caller's voice; a voiceless animal asks a partner close by that has
/// one to call for it.
pub(super) fn call(
    agent: Agent,
    relations: &Relations,
    index: &HashMap<Entity, Agent>,
    calls: &mut Vec<Call>,
) {
    let kind = agent.kind;
    let voice = kind.voice();
    if voice > 0.0 {
        calls.push(Call {
            caller: kind,
            at: agent.position,
            reach: voice,
        });
    } else if let Some(partner) = relations
        .companion
        .and_then(|partner| index.get(&partner))
        .filter(|partner| partner.position.distance(agent.position) < 80.0)
        && let Some(voice) = partner.kind.animal().map(|animal| animal.voice)
        && voice > 0.0
    {
        calls.push(Call {
            caller: partner.kind,
            at: partner.position,
            reach: voice,
        });
    }
}

/// A hunter at its prey: holding what it has caught, by web or by latching on, and dealing its blow
/// when the prey is in reach and it is ready.
pub(super) fn engage(
    agent: Agent,
    prey: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    index: &HashMap<Entity, Agent>,
    (attacks, now, dt): (&mut Vec<Attack>, f32, f32),
) {
    let biology = agent.kind.biology();
    let ethology = agent.kind.behaviour();
    let activity = relations.activity;
    let distance = prey.position.distance(agent.position);
    let reach = reach(agent.kind, prey.kind);
    // Prey the hunter holds, in a web or by latching on.
    let web = relations.home.and_then(|home| index.get(&home));
    let snare = Snare {
        distance,
        from_web: web.map(|web| prey.position.distance(web.position)),
        reach,
        share: (biology.adult_mass / prey.kind.biology().adult_mass).min(1.0),
        fastened: prey.vitals.stunned > 0.0,
        dt,
    };
    if let Some(hold) = (ethology.snare)(&snare)
        && (!hold.latch || activity == Activity::Attach)
    {
        if hold.latch {
            relations.host = Some(prey.entity);
        }
        attacks.push(Attack {
            victim: prey.entity,
            strike: (hold.drain > 0.0).then_some(Strike {
                from: SceneVec3::ZERO,
                to: SceneVec3::ZERO,
                radius: 0.0,
                force: hold.drain,
                injury: Injury::Drain,
            }),
            stun: hold.stun,
            carrier: None,
        });
    }
    if activity == Activity::Strike && distance < reach + ethology.range {
        blow(agent, prey, body, relations, (attacks, now));
    }
}

/// How near two organisms must be for one to reach the other: their sizes, and a little more.
fn reach(one: Kind, other: Kind) -> f32 {
    one.biology().adult_mass.cbrt() * 2.0 + other.biology().adult_mass.cbrt() * 2.0 + 3.0
}

/// Deal a blow to `victim` in the animal's own way, if it is ready to.
fn blow(
    agent: Agent,
    victim: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    (attacks, now): (&mut Vec<Attack>, f32),
) {
    let ethology = agent.kind.behaviour();
    if now < relations.ready {
        return;
    }
    let Some(blow) = (ethology.blow)(&aim(agent, victim, body, relations.mind.wear)) else {
        return;
    };
    body.reserve = (body.reserve - blow.cost).max(0.0);
    // A weapon driven into something too heavy to fly with would pin the hunter to it: it sheds
    // the weapon's head to get free, and the iron it was made of goes with it.
    if ethology.skewers && victim.vitals.mass > agent.vitals.mass * ethology.largest * 1.5 {
        body.iron = (body.iron - 0.25).max(0.0);
    }
    relations.mind.wear = (relations.mind.wear + 0.08).min(1.0);
    // Every blow it lands teaches it a little more.
    relations.mind.experience.succeeded();
    attacks.push(Attack {
        victim: victim.entity,
        strike: Some(blow.strike),
        stun: blow.stun,
        carrier: None,
    });
    relations.ready = now + blow.rest;
}

/// Where the mind meant to go this step, in the world.
fn aimed_at(agent: Agent, relations: &Relations) -> Option<Vec3> {
    let offset = relations.senses.goal?.offset?;
    Some(agent.position + offset.to_engine())
}

/// The neighbour nearest `point` that `fits`.
fn nearest(neighbours: &[Agent], point: Vec3, fits: impl Fn(&Agent) -> bool) -> Option<Agent> {
    neighbours
        .iter()
        .filter(|other| fits(other))
        .min_by(|a, b| {
            a.position
                .distance_squared(point)
                .total_cmp(&b.position.distance_squared(point))
        })
        .copied()
}

/// Turning on a hunter: whatever it was driving at, if it is a hunter and in reach, is dealt with.
///
/// A guardian keeping a sanctuary harms no one in it: it shoulders the hunter out, away from the
/// young, and its stand-off breaks the hunt. Any other animal defending itself takes to the hunter
/// with the blow it would deal its prey.
fn defend(agent: Agent, body: &mut Vitals, relations: &mut Relations, step: &mut Conduct) {
    let Some(at) = aimed_at(agent, relations) else {
        return;
    };
    let foe = nearest(step.neighbours, at, |other| {
        other.entity != agent.entity
            && other.kind != agent.kind
            && other.alive()
            && !other.protected
            && other.kind.takes(Take::Prey)
    });
    if let Some(foe) = foe.filter(|foe| {
        foe.position.distance(agent.position)
            < reach(agent.kind, foe.kind) + agent.kind.behaviour().range
    }) {
        if agent.kind.behaviour().shelters > 0.0 {
            let away = (foe.position - agent.position).normalize_or_zero();
            step.nudges.push((foe.entity, away * 6.0));
        } else {
            blow(agent, foe, body, relations, (step.attacks, step.now));
        }
    }
}

/// Binding the jungle: a Weaver at work on its web ties in any plant drifting near its home that
/// nothing holds yet, drawing it a little closer, so over time its home gathers a loose mass of
/// plants of every kind about it, joined wherever two came near.
fn bind(agent: Agent, relations: &Relations, step: &mut Conduct) {
    let Some(home) = relations.home else {
        return;
    };
    let Some(anchor) = step.neighbours.iter().find(|other| other.entity == home) else {
        return;
    };
    let reach = anchor.kind.render_radius() * 2.5 + 40.0;
    let stray = nearest(step.neighbours, anchor.position, |other| {
        other.entity != home
            && other.entity != agent.entity
            && plant(other.kind)
            && other.alive()
            && other.tether.is_none()
            && !other.kind.attaches()
    })
    .filter(|stray| stray.position.distance(anchor.position) < reach);
    if let Some(stray) = stray {
        let touching = (stray.kind.render_radius() + anchor.kind.render_radius()) * 0.6;
        let rest = (stray.position.distance(anchor.position) * 0.7).max(touching);
        step.ties.push((stray.entity, home, rest));
    }
}

/// Pushing something along: an egg or a young one, or a plant, in reach is moved the way the
/// animal is pushing, the more readily the lighter it is against the one pushing.
fn nudge(agent: Agent, relations: &Relations, step: &mut Conduct) {
    let Some(at) = aimed_at(agent, relations) else {
        return;
    };
    let ward = nearest(step.neighbours, agent.position, |other| {
        other.entity != agent.entity
            && other.alive()
            && !other.protected
            && (plant(other.kind) || other.grown() < 0.5)
    })
    .filter(|ward| ward.position.distance(agent.position) < reach(agent.kind, ward.kind));
    if let Some(ward) = ward {
        let way = (at - ward.position).normalize_or_zero();
        let leverage = (agent.vitals.mass / ward.vitals.mass.max(1e-6) * 20.0).min(1.0);
        step.nudges.push((ward.entity, way * 1.5 * leverage));
    }
}

/// Bringing a guardian food: some of what is in its gut is handed over while it is close.
fn offer(agent: Agent, body: &mut Vitals, step: &mut Conduct) {
    let guardian = nearest(step.neighbours, agent.position, |other| {
        other.alive() && other.grown() >= 1.0 && other.kind.behaviour().shelters > 0.0
    })
    .filter(|guardian| {
        guardian.position.distance(agent.position) < reach(agent.kind, guardian.kind) + 3.0
    });
    if let Some(guardian) = guardian {
        let gift = (body.mass * step.dt * 0.003).min(body.gut * 0.5);
        if gift > 0.0 {
            body.gut -= gift;
            step.gifts.push((guardian.entity, gift));
        }
    }
}

/// Mist-script is written in breath: each glyph spends water, and a little aur to light it.
pub(super) fn write(body: &mut Vitals, dt: f32) {
    body.hydration = (body.hydration - 0.0006 * dt).max(0.0);
    body.aur = (body.aur - 0.0002 * dt).max(0.0);
}

/// The step's bookkeeping shared by every animal's conduct: the organisms about it, and the blows,
/// calls, pushes and gifts it gives others, settled once all have stepped.
pub(super) struct Conduct<'a> {
    pub(super) now: f32,
    pub(super) dt: f32,
    pub(super) neighbours: &'a [Agent],
    pub(super) attacks: &'a mut Vec<Attack>,
    pub(super) calls: &'a mut Vec<Call>,
    pub(super) nudges: &'a mut Vec<(Entity, Vec3)>,
    pub(super) gifts: &'a mut Vec<(Entity, f32)>,
    /// Plants tied to others by silk, the anchor each is tied to, and the length of the tie.
    pub(super) ties: &'a mut Vec<(Entity, Entity, f32)>,
}

/// Carry out what the mind chose: strike prey within reach, string and tend a web, latch onto a
/// host, court a mate to the end, call, sleep and cling to a roost.
pub(super) fn conduct(
    agent: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    targets: &Targets,
    index: &HashMap<Entity, Agent>,
    mut step: Conduct,
) {
    let kind = agent.kind;
    let ethology = kind.behaviour();
    let activity = relations.activity;
    match activity {
        Activity::Weave => bind(agent, relations, &mut step),
        Activity::Defend => defend(agent, body, relations, &mut step),
        Activity::Nudge => nudge(agent, relations, &mut step),
        Activity::Offer => offer(agent, body, &mut step),
        _ => {}
    }
    let Conduct {
        now,
        dt,
        attacks,
        calls,
        ..
    } = step;
    if let Some(prey) = targets.prey {
        engage(agent, prey, body, relations, index, (attacks, now, dt));
    }
    attacks.extend(carry(agent, targets.prey, activity));
    // A feeder fastened on a host lets go once it turns to anything else, unless its lineage
    // lives fastened.
    if relations.host.is_some()
        && relations.host == targets.prey.map(|prey| prey.entity)
        && !matches!(activity, Activity::Attach | Activity::Feed)
        && !kind.attaches()
    {
        relations.host = None;
    }
    // Prey held only by venom or silk escapes once it wears off.
    if relations.captor.is_some() && body.stunned <= 0.0 {
        relations.host = None;
        relations.captor = None;
    }
    if ethology.anchored && relations.home.is_none() {
        relations.home = targets.roost;
    }
    // Courting: a pair displaying to one another long enough are mated.
    if activity == Activity::Display {
        let together = targets.mate.is_some_and(|mate| {
            mate.position.distance(agent.position) < 15.0 && mate.activity == Activity::Display
        });
        if together {
            relations.mind.courted += dt;
            if relations.mind.courted > 8.0 {
                relations.fertile_until = now + 900.0;
                relations.sire = targets.mate.map(|mate| mate.genes);
                relations.mind.courted = 0.0;
            }
        }
    }
    if relations.senses.goal.is_some_and(|goal| goal.call) {
        call(agent, relations, index, calls);
    }
    // Honing brings back the edge its blows wore away.
    if activity == Activity::Hone {
        relations.mind.wear = (relations.mind.wear - 0.05 * dt).max(0.0);
    }
    if activity == Activity::Sign {
        write(body, dt);
    }
    if activity == Activity::Sleep {
        body.sleep(dt);
        // Animals that roost cling to the plant they sleep on.
        if ethology.clings
            && let Some(roost) = targets.roost.and_then(|roost| index.get(&roost))
            && roost.position.distance(agent.position) < 12.0
        {
            relations.host = Some(roost.entity);
            if !ethology.anchored {
                relations.home = Some(roost.entity);
            }
        }
    } else if relations.host.is_some()
        && relations.host == relations.home
        && relations.parent.is_none()
        && !kind.attaches()
    {
        relations.host = None;
    }
}

/// A hunter that carries its kill holds helpless prey in its mouth tentacles while it eats.
pub(super) fn carry(hunter: Agent, prey: Option<Agent>, activity: Activity) -> Option<Attack> {
    let prey = prey?;
    let close = prey.position.distance(hunter.position)
        < hunter.kind.biology().adult_mass.cbrt() * 2.0 + 4.0;
    let ethology = hunter.kind.behaviour();
    let light =
        !ethology.skewers || prey.vitals.mass <= hunter.vitals.mass * ethology.largest * 1.5;
    (ethology.carries && light && activity == Activity::Feed && prey.vitals.stunned > 0.0 && close)
        .then_some(Attack {
            victim: prey.entity,
            strike: None,
            stun: 1.0,
            carrier: Some(hunter.entity),
        })
}

/// How a hunter's blow lands, seen from inside its prey: the way the hunter is moving and where it
/// is in the prey's frame, the prey's size, how far wide it goes for how the prey slips across its
/// path and how sure the hunter's aim is, where the prey's pump lies, and the hunter's reserve.
pub(super) fn aim(hunter: Agent, prey: Agent, body: &Vitals, wear: f32) -> Aim {
    let viscera = prey.kind.organs();
    let extent = viscera.extent();
    let turn = prey.rotation.inverse();
    let local = |world: Vec3| (turn * world).to_scene();
    let along = local(
        hunter
            .velocity
            .normalize_or(prey.position - hunter.position),
    )
    .normalize_or(SceneVec3::NEG_Z);
    let toward = local(hunter.position - prey.position).normalize_or(SceneVec3::Y);
    // A blow goes wide the way the prey is slipping across its path, the further the quicker and
    // more agile the prey and the less sure the hunter's aim; a sitting target is struck squarely.
    let slip = local(prey.velocity - hunter.velocity);
    let across = (slip - along * slip.dot(along)).normalize_or_zero();
    let dodge = (0.25 + (prey.velocity.length() / prey.kind.speed().max(1.0)).min(1.0) * 0.35)
        * (0.6 + prey.genes.get(AGILITY) * 0.8);
    let sure = (hunter.genes.get(PRECISION) * 0.6 + hunter.skill * 0.4).clamp(0.0, 1.0);
    Aim {
        along,
        toward,
        extent,
        scatter: across * extent * dodge * (1.0 - sure * 0.7),
        pump: viscera.find(Function::Circulation).unwrap_or_default(),
        nerve: viscera.find(Function::Coordination).unwrap_or_default(),
        reserve: body.reserve,
        edge: 1.0 - wear * 0.6,
    }
}

/// Land the step's blows on their victims, and count the dead.
pub(super) fn land(organisms: &mut Query<Organisms>, attacks: &[Attack]) -> u64 {
    let mut deaths = 0;
    for attack in attacks {
        let Ok((_, species, mut body, mut relations, _, _, player, mount, _)) =
            organisms.get_mut(attack.victim)
        else {
            continue;
        };
        if body.0.health <= 0.0 || player || mount {
            continue;
        }
        body.0.stunned = body.0.stunned.max(attack.stun);
        if let Some(carrier) = attack.carrier {
            relations.host = Some(carrier);
            relations.captor = Some(carrier);
        }
        if let Some(strike) = attack.strike {
            let extent = species.0.organs().extent().length();
            let injury = body.0.wound(species.0, strike, extent);
            if body.0.health <= 0.0 {
                deaths += 1;
            } else {
                // Living through it leaves it warier.
                relations.mind.experience.survived(injury.damage);
            }
        }
    }
    deaths
}
