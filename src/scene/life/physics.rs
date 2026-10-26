//! How bodies move through the air.
//!
//! Every body, plant or animal, is pulled down by the planet's light gravity and pushed by the air
//! it moves through. Gas stores bear some or all of its weight; the air drags it towards its own
//! motion, so the updraft lifts what drifts in it and the wind carries it along; and a wing moving
//! through the air at speed bears the weight it would otherwise have to beat for. What is left an
//! animal makes up with its wings, as hard as its muscles, its breath and the state of its wings
//! let it, and a floater by filling or venting its gas, as far as it can; how hard that was is the
//! effort it spends, and the force it leans on is what tilts it into its turns.

use crate::scene::{
    life::{Kind, Kingdom, organs::Function},
    planet::GRAVITY,
};
use spirv_std::glam::Vec3;

/// How a body of a kind meets the air, each per unit of its mass.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// How hard the air drags on it for its weight: the small and the broad-leaved far more than
    /// the large and compact.
    pub drag: f32,
    /// How much of its weight its gas stores bear when full.
    pub buoyant: f32,
    /// How far it can fill or vent its gas to rise or sink, as an acceleration.
    pub trim: f32,
    /// The most its wings or fins can push, as an acceleration.
    pub thrust: f32,
    /// The airspeed at which its wings bear its whole weight without beating; zero for a body with
    /// no wings to glide on.
    pub glide: f32,
}

/// The air's drag at a standstill, as if moving this fast, so a body at rest in the air is still
/// drawn along by it.
const STILL: f32 = 3.0;

impl Kind {
    /// How its body meets the air, from its size and what it is built with.
    pub fn body(self) -> Body {
        let length = self.render_radius().max(0.05);
        let organs = self.organs();
        let gas = organs.has(Function::Lift);
        // Whatever lives fastened to a host is carried by it; loose, between hosts, it is a
        // spore-light thing that rides the updraft, nearly weightless in the air, until it finds
        // another.
        if self.attaches() {
            return Body {
                drag: 0.07 / length,
                buoyant: 0.95,
                trim: GRAVITY * 0.1,
                thrust: if self.animal().is_some() {
                    GRAVITY * 0.6
                } else {
                    0.0
                },
                glide: 0.0,
            };
        }
        match (self.kingdom(), self.animal()) {
            (Kingdom::Plantae, _) => Body {
                // Fronds, sails and roots catch the air far more than any animal of their size.
                drag: 0.07 / length,
                buoyant: if self.floats() { 1.0 } else { 0.0 },
                trim: if self.floats() { GRAVITY * 0.25 } else { 0.0 },
                thrust: 0.0,
                glide: 0.0,
            },
            (_, Some(animal)) if animal.flies() => Body {
                // Streamlined for flight, they slip through the air far more easily.
                drag: 0.003 / length,
                buoyant: if gas { 0.35 } else { 0.0 },
                trim: if gas { GRAVITY * 0.05 } else { 0.0 },
                thrust: GRAVITY * 2.4,
                // The speed it glides best at, well below its fastest.
                glide: animal.speed * 0.35,
            },
            (_, Some(animal)) if animal.floats => Body {
                // Gas-bag animals are all lift and drag, and push only gently.
                drag: 0.04 / length,
                buoyant: 1.0,
                trim: GRAVITY * 0.3,
                thrust: GRAVITY * 0.4,
                glide: 0.0,
            },
            _ => Body {
                drag: 0.02 / length,
                buoyant: 0.0,
                trim: 0.0,
                thrust: GRAVITY * 0.6,
                glide: 0.0,
            },
        }
    }
}

/// The state a body is in as it moves: how full its gas stores are, how hard it can work, how
/// whole its wings are and which side is the weaker.
#[derive(Clone, Copy, Debug)]
pub struct Condition {
    pub gas: f32,
    pub power: f32,
    pub wings: f32,
    /// From the left wings weaker (below zero) to the right (above).
    pub list: f32,
}

/// What the air and the body's own efforts do to it this moment.
#[derive(Clone, Copy, Debug, Default)]
pub struct Forces {
    pub acceleration: Vec3,
    /// How hard it is working its wings, as a share of all they can give.
    pub effort: f32,
    /// What holds it up and turns it, besides its gas: the lift of its wings, beaten or gliding.
    pub support: Vec3,
}

/// The forces on a body moving at `velocity` through air moving at `wind`, when it wants to
/// accelerate at `wanted`, with `up` away from the planet and `right` to its right.
pub fn forces(
    body: Body,
    condition: Condition,
    velocity: Vec3,
    wind: Vec3,
    up: Vec3,
    right: Vec3,
    wanted: Vec3,
) -> Forces {
    let air = velocity - wind;
    let weight = -up * GRAVITY;
    let buoyancy = up * GRAVITY * body.buoyant * condition.gas.clamp(0.0, 1.0);
    let drag = -air * body.drag * (air.length() + STILL);
    // Gliding lift, growing with the square of the airspeed across the wings, up to all of the
    // weight the gas does not bear.
    let level = air - up * air.dot(up);
    let unborne = (GRAVITY - buoyancy.dot(up)).max(0.0);
    let gliding = if body.glide > 0.0 {
        let share = (level.length_squared() / (body.glide * body.glide)).min(1.0);
        up * unborne * share * condition.wings
    } else {
        Vec3::ZERO
    };
    let passive = weight + buoyancy + drag + gliding;
    let needed = wanted - passive;
    let trim = up * needed.dot(up).clamp(-body.trim, body.trim);
    let most = body.thrust * condition.power.clamp(0.0, 1.0) * condition.wings;
    let mut beaten = needed - trim;
    let length = beaten.length();
    if length > most {
        beaten *= most / length.max(1e-6);
    }
    // Weaker wings on one side give less on that side, and the body slips towards it.
    let beaten = beaten - right * condition.list * beaten.length() * 0.2;
    Forces {
        acceleration: passive + trim + beaten,
        // Damaged wings must work the harder for the same push.
        effort: if body.thrust > 0.0 {
            beaten.length() / (body.thrust * condition.wings.max(0.2))
        } else {
            0.0
        },
        support: gliding + beaten,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::KINDS;

    const CALM: Condition = Condition {
        gas: 1.0,
        power: 1.0,
        wings: 1.0,
        list: 0.0,
    };

    #[test]
    fn every_flier_can_hover_and_glides_cheaper_than_it_hovers() {
        for kind in KINDS
            .iter()
            .copied()
            .filter(|kind| kind.animal().is_some_and(|animal| animal.flies()))
        {
            let body = kind.body();
            let hover = forces(
                body,
                CALM,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::Y,
                Vec3::X,
                Vec3::ZERO,
            );
            assert!(hover.acceleration.length() < 1e-3, "{kind:?} cannot hover");
            let cruise = Vec3::Z * body.glide;
            let gliding = forces(body, CALM, cruise, Vec3::ZERO, Vec3::Y, Vec3::X, Vec3::ZERO);
            assert!(
                gliding.effort < hover.effort,
                "{kind:?} works harder gliding than hovering"
            );
        }
    }

    #[test]
    fn plants_fall_without_their_gas_and_drift_with_the_wind() {
        let body = Kind::Laevivelum.body();
        let empty = Condition { gas: 0.0, ..CALM };
        let falling = forces(
            body,
            empty,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::X,
            Vec3::ZERO,
        );
        assert!(
            falling.acceleration.y < -0.1,
            "an empty plant does not fall"
        );
        let wind = Vec3::X * 5.0;
        let carried = forces(body, CALM, Vec3::ZERO, wind, Vec3::Y, Vec3::X, Vec3::ZERO);
        assert!(
            carried.acceleration.x > 0.0,
            "the wind does not carry a plant"
        );
    }
}
