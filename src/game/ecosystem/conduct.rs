//! What an animal does once it has decided: strike and hold its prey, court, call and sleep.

use crate::{
    game::ecosystem::{Agent, Attack, Call, Organisms, Relations, senses::Targets},
    math::ToScene,
};
use bevy::prelude::*;
use skybound::scene::life::{
    animalia::{
        AGILITY, PRECISION,
        mind::{Aim, Snare},
    },
    behaviour::Activity,
    organs::{Function, Strike, Wound},
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
    let reach = biology.adult_mass.cbrt() * 2.0 + prey.kind.biology().adult_mass.cbrt() * 2.0 + 3.0;
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
                wound: Wound::Drain,
            }),
            stun: hold.stun,
            carrier: None,
        });
    }
    if activity == Activity::Strike
        && distance < reach + ethology.range
        && now >= relations.ready
        && let Some(blow) = (ethology.blow)(&aim(agent, prey, body, relations.mind.wear))
    {
        body.reserve = (body.reserve - blow.cost).max(0.0);
        relations.mind.wear = (relations.mind.wear + 0.08).min(1.0);
        // Every blow it lands teaches it a little more.
        relations.mind.experience.succeeded();
        attacks.push(Attack {
            victim: prey.entity,
            strike: Some(blow.strike),
            stun: blow.stun,
            carrier: None,
        });
        relations.ready = now + blow.rest;
    }
}

/// Mist-script is written in breath: each glyph spends water, and a little aur to light it.
pub(super) fn write(body: &mut Vitals, dt: f32) {
    body.hydration = (body.hydration - 0.0006 * dt).max(0.0);
    body.aur = (body.aur - 0.0002 * dt).max(0.0);
}

/// The step's bookkeeping shared by every animal's conduct.
pub(super) struct Conduct<'a> {
    pub(super) now: f32,
    pub(super) dt: f32,
    pub(super) attacks: &'a mut Vec<Attack>,
    pub(super) calls: &'a mut Vec<Call>,
}

/// Carry out what the mind chose: strike prey within reach, string and tend a web, latch onto a
/// host, court a mate to the end, call, sleep and cling to a roost.
pub(super) fn conduct(
    agent: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    targets: &Targets,
    index: &HashMap<Entity, Agent>,
    step: Conduct,
) {
    let Conduct {
        now,
        dt,
        attacks,
        calls,
    } = step;
    let kind = agent.kind;
    let ethology = kind.behaviour();
    let activity = relations.activity;
    if let Some(prey) = targets.prey {
        engage(agent, prey, body, relations, index, (attacks, now, dt));
    }
    attacks.extend(carry(agent, targets.prey, activity));
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
    (hunter.kind.behaviour().carries
        && activity == Activity::Feed
        && prey.vitals.stunned > 0.0
        && close)
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
