use spirv_std::glam::{Mat3, Quat, Vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::Kind;
use super::biology::{Anatomy, Biology, Feeding, Social};

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Activity {
    #[default]
    Drift,
    Feed,
    Flee,
    Drink,
    Rest,
    Return,
    Attach,
    Clean,
    Weave,
    Guard,
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

pub fn drift(s: &Senses, lift: f32) -> Controls {
    Controls {
        activity: Activity::Drift,
        velocity: s.wind + s.up.normalize_or_zero() * (lift + s.buoyancy),
        facing: s.wind.normalize_or_zero(),
        response: 0.35,
        wing_fold: 0.0,
    }
}

pub fn organism(s: &Senses, biology: Biology) -> Controls {
    if biology.anatomy == Anatomy::Porous {
        return drift(s, 0.0);
    }
    let (target, speed, activity) = match (s.danger, s.water, s.habitat) {
        (Some(danger), _, _) => {
            if biology.social == Social::Guardian {
                (Some(danger), biology.speed, Activity::Guard)
            } else {
                (Some(-danger), biology.speed * 1.2, Activity::Flee)
            }
        }
        (_, Some(water), _) if s.hydration < 0.35 => {
            (Some(water), biology.speed * 0.7, Activity::Drink)
        }
        (_, _, Some(habitat)) => (Some(habitat), biology.speed * 0.6, Activity::Return),
        _ if s.fatigue > 0.8 => (None, 0.0, Activity::Rest),
        _ if s.energy < 0.25 && s.food.is_some() => (s.food, biology.speed * 0.7, Activity::Feed),
        _ if biology.social == Social::Mutualist && s.companion.is_some() => {
            (s.companion, biology.speed * 0.6, Activity::Attach)
        }
        _ if biology.feeding == Feeding::Cleaner && s.food.is_some() => {
            (s.food, biology.speed * 0.6, Activity::Clean)
        }
        _ if s.energy < 0.85 && s.food.is_some() => (s.food, biology.speed * 0.7, Activity::Feed),
        _ if biology.social == Social::Weaver && s.companion.is_some() => {
            (s.companion, biology.speed * 0.3, Activity::Weave)
        }
        _ => (s.companion, biology.speed * 0.25, Activity::Drift),
    };
    let mut controls = if biology.anatomy == Anatomy::Plant {
        drift(s, 0.0)
    } else {
        let mut controls = steer(s, target, speed);
        controls.velocity += s.separation * 2.0 + s.alignment * 0.2;
        if target.is_some_and(|offset| offset.length_squared() < 9.0) {
            controls.velocity = s.wind + s.separation;
        }
        controls.wing_fold = (-controls.facing.dot(s.up)).clamp(0.0, 1.0);
        controls
    };
    controls.activity = activity;
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

#[derive(Clone, Copy)]
pub struct Motion {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
}

impl Motion {
    pub fn advance(&mut self, kind: Kind, senses: &Senses, seconds: f32) -> Controls {
        let controls = super::behave(kind, senses);
        let dt = seconds.max(0.0);
        let response = controls.response.max(0.001);
        let decay = (-response * dt).exp();
        self.position += controls.velocity * dt
            + (self.velocity - controls.velocity) * ((1.0 - decay) / response);
        self.velocity = controls.velocity + (self.velocity - controls.velocity) * decay;
        let forward = controls.facing.normalize_or_zero();
        let up = senses.up.normalize_or_zero();
        let right = forward.cross(up).normalize_or_zero();
        if right.length_squared() > 0.5 {
            let rotation = Quat::from_mat3(&Mat3::from_cols(right, right.cross(forward), -forward));
            self.rotation = self.rotation.slerp(rotation, 1.0 - decay);
        }
        controls
    }
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
        for kind in [Kind::Aerogigans, Kind::Levita, Kind::Nubiballona] {
            let passive = super::super::behave(kind, &senses);
            let empty = super::super::behave(
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
