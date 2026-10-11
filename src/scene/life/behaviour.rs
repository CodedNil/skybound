use spirv_std::glam::{Mat3, Quat, Vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::Kind;

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
            aur: 0.5,
            time: 0.0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Controls {
    pub velocity: Vec3,
    pub facing: Vec3,
    pub response: f32,
    pub wing_fold: f32,
}

pub fn drift(s: &Senses, lift: f32) -> Controls {
    Controls {
        velocity: s.wind + s.up.normalize_or_zero() * lift,
        facing: s.wind.normalize_or_zero(),
        response: 0.35,
        wing_fold: 0.0,
    }
}

pub fn forage(s: &Senses, speed: f32) -> Controls {
    let direction = s
        .danger
        .map_or_else(|| s.food.unwrap_or(s.wind), |danger| -danger)
        .normalize_or_zero();
    let up = s.up.normalize_or_zero();
    Controls {
        velocity: s.wind + direction * speed,
        facing: direction,
        response: 1.8,
        wing_fold: (-direction.dot(up)).clamp(0.0, 1.0),
    }
}

pub fn seek_aur(s: &Senses, speed: f32) -> Controls {
    let mut controls = drift(s, (s.aur - 0.5) * speed);
    controls.facing = controls.velocity.normalize_or_zero();
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
