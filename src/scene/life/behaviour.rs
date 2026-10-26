use crate::scene::life::{Kind, behave};
use spirv_std::glam::{Mat3, Quat, Vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// What an organism is visibly doing this moment, which its pose and the others around it
/// respond to.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Activity {
    /// Going where the air and its kind take it.
    #[default]
    Drift,
    /// Eating.
    Feed,
    /// Running from danger.
    Flee,
    /// Drinking at a Pondplant.
    Drink,
    /// Resting, wings half folded.
    Rest,
    /// Making its way back to where it belongs.
    Return,
    /// Fastened onto a host or companion.
    Attach,
    /// Cleaning a host of parasites.
    Clean,
    /// Tending its web.
    Weave,
    /// Defending the young, or harrying a predator.
    Guard,
    /// Asleep, wings folded.
    Sleep,
    /// Stalking or closing on prey.
    Hunt,
    /// Delivering the blow: the stoop, the pass, the lunge or the shot.
    Strike,
    /// Courting, crest or lights or song on show.
    Display,
    /// Looking over something strange.
    Investigate,
    /// Chasing its fellows in play.
    Play,
    /// Grooming or greeting one of its fellows, close alongside.
    Groom,
    /// Drawing a blade along something hard to keep its edge.
    Hone,
    /// Speaking in mist-script, as the voiceless Næ do: breathing out glyphs of vapour lit from
    /// within by its aur sacs, toned by the colour of its crest.
    Sign,
}

/// What a mind has decided to do this moment, for the body to carry out.
///
/// It names the activity, where to go as an offset from where it is, how fast as a share of
/// its top speed, how far to fold its wings, and whether to hold station on arrival rather
/// than fly through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Goal {
    pub activity: Activity,
    pub offset: Option<Vec3>,
    pub speed: f32,
    pub fold: f32,
    pub hover: bool,
    /// Whether it calls as it goes, a call its kin hear from far off.
    pub call: bool,
}

impl Goal {
    pub const fn new(activity: Activity, offset: Option<Vec3>, speed: f32) -> Self {
        Self {
            activity,
            offset,
            speed,
            fold: 0.0,
            hover: false,
            call: false,
        }
    }

    /// The same goal, calling as it goes.
    pub const fn calling(self) -> Self {
        Self { call: true, ..self }
    }

    /// The same goal, holding station on arrival.
    pub const fn hovering(self) -> Self {
        Self {
            hover: true,
            ..self
        }
    }

    /// The same goal, with the wings folded this far.
    pub const fn folded(self, fold: f32) -> Self {
        Self { fold, ..self }
    }
}

#[derive(Clone, Copy)]
pub struct Senses {
    pub wind: Vec3,
    pub up: Vec3,
    /// Offsets from the organism to nearby targets.
    pub food: Option<Vec3>,
    pub danger: Option<Vec3>,
    pub companion: Option<Vec3>,
    pub habitat: Option<Vec3>,
    pub storm: Option<Vec3>,
    pub water: Option<Vec3>,
    pub separation: Vec3,
    pub alignment: Vec3,
    pub energy: f32,
    pub hydration: f32,
    pub fatigue: f32,
    pub buoyancy: f32,
    pub aur: f32,
    pub time: f32,
    /// What the animal's mind has decided, when it has one.
    pub goal: Option<Goal>,
    /// How hard its body can work, as a share of a whole one's.
    pub power: f32,
    /// How far it is pulled down by gas lost from its stores.
    pub sinking: f32,
}

impl Default for Senses {
    fn default() -> Self {
        Self {
            wind: Vec3::ZERO,
            up: Vec3::Y,
            food: None,
            danger: None,
            companion: None,
            habitat: None,
            storm: None,
            water: None,
            separation: Vec3::ZERO,
            alignment: Vec3::ZERO,
            energy: 0.5,
            hydration: 1.0,
            fatigue: 0.0,
            buoyancy: 0.0,
            aur: 0.5,
            time: 0.0,
            goal: None,
            power: 1.0,
            sinking: 0.0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Controls {
    pub activity: Activity,
    pub velocity: Vec3,
    pub facing: Vec3,
    pub response: f32,
    pub wing_fold: f32,
}

#[derive(Clone, Copy)]
pub struct Motion {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
}

impl Motion {
    pub fn advance(&mut self, kind: Kind, senses: &Senses, seconds: f32) -> Controls {
        let controls = behave(kind, senses);
        let dt = seconds.max(0.0);
        let response = controls.response.max(0.001);
        let decay = (-response * dt).exp();
        self.position += controls.velocity * dt
            + (self.velocity - controls.velocity) * ((1.0 - decay) / response);
        self.velocity = controls.velocity + (self.velocity - controls.velocity) * decay;
        let up = senses.up.normalize_or_zero();
        // A plant holds itself upright on its lift; an animal pitches into a climb or a dive, but
        // never further than its wings can carry it, so none hangs on its side or stands on end.
        let steepest = if kind.animal().is_some() { 0.6 } else { 0.0 };
        let forward = level(controls.facing, up, steepest);
        let right = forward.cross(up).normalize_or_zero();
        if right.length_squared() > 0.5 {
            let rotation = Quat::from_mat3(&Mat3::from_cols(right, right.cross(forward), -forward));
            self.rotation = self.rotation.slerp(rotation, 1.0 - decay);
        }
        controls
    }
}

/// A heading pitched no more steeply than `steepest`, the sine of the angle from the horizon.
fn level(facing: Vec3, up: Vec3, steepest: f32) -> Vec3 {
    let facing = facing.normalize_or_zero();
    let rise = facing.dot(up);
    // Facing straight up or down gives no heading to keep, so it keeps the one it has.
    let across = (facing - up * rise).normalize_or_zero();
    if across == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let rise = rise.clamp(-steepest, steepest);
    across * (1.0 - rise * rise).sqrt() + up * rise
}

pub fn drift(s: &Senses, lift: f32) -> Controls {
    Controls {
        activity: Activity::Drift,
        velocity: s.wind + s.up.normalize_or_zero() * (lift + s.buoyancy),
        facing: s.wind.normalize_or_zero(),
        response: 0.35,
        wing_fold: 0.0,
    }
}

/// How an organism moves this step. A plant drifts; an animal pursues what its mind decided, or,
/// with no mind at work (an egg, a newborn, or the player's mount), keeps with its companion.
pub fn organism(s: &Senses, kind: Kind) -> Controls {
    let Some(animal) = kind.animal() else {
        return drift(s, -s.sinking);
    };
    if let Some(goal) = s.goal {
        return pursue(s, animal.speed, goal);
    }
    let mut controls = steer(s, s.companion.or(s.habitat), animal.speed * 0.4);
    controls.velocity += s.separation * 2.0 + s.alignment * 0.2;
    if s.companion
        .is_some_and(|offset| offset.length_squared() < 9.0)
    {
        controls.velocity = s.wind + s.separation;
    }
    controls.wing_fold = (-controls.facing.dot(s.up)).clamp(0.0, 1.0);
    controls
}

/// Carry out a mind's goal: fly towards its target at its pace, held back by a weakened body,
/// slowing to hold station if it is to hover there.
fn pursue(s: &Senses, top: f32, goal: Goal) -> Controls {
    let speed = top * goal.speed * s.power;
    let mut controls = steer(s, goal.offset, speed);
    controls.velocity -= s.up.normalize_or_zero() * s.sinking;
    if let Some(offset) = goal.offset {
        let distance = offset.length();
        if goal.hover && distance < 6.0 {
            controls.velocity = s.wind + offset * 0.5 + s.separation;
        }
    } else if goal.hover {
        controls.velocity = s.wind * 0.3 + s.separation;
    }
    if goal.activity != Activity::Strike {
        controls.velocity += s.separation * 2.0;
    }
    controls.activity = goal.activity;
    controls.wing_fold = goal.fold.max((-controls.facing.dot(s.up)).clamp(0.0, 1.0));
    controls.response = if goal.activity == Activity::Strike {
        2.5
    } else {
        controls.response
    };
    controls
}

pub fn steer(s: &Senses, target: Option<Vec3>, speed: f32) -> Controls {
    let mut controls = drift(s, 0.0);
    if let Some(offset) = target {
        controls.facing = offset.normalize_or_zero();
        controls.velocity += controls.facing * speed;
    }
    controls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passive_feeders_ignore_targets_and_motion_is_frame_independent() {
        let senses = Senses {
            wind: Vec3::new(4.0, 1.0, -2.0),
            food: Some(Vec3::X),
            danger: Some(Vec3::Z),
            ..Senses::default()
        };
        for kind in [Kind::Aerogigas, Kind::Levitans, Kind::Nubivesica] {
            let passive = behave(kind, &senses);
            let empty = behave(
                kind,
                &Senses {
                    food: None,
                    danger: None,
                    ..senses
                },
            );
            assert_eq!(
                passive.velocity, empty.velocity,
                "passive feeder chased a target"
            );
            let mut whole = Motion {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            };
            let mut split = whole;
            whole.advance(kind, &senses, 1.0);
            for _ in 0..60 {
                split.advance(kind, &senses, 1.0 / 60.0);
            }
            assert!(
                whole.position.distance(split.position) < 1e-4,
                "frame-dependent motion"
            );
        }
    }
}
