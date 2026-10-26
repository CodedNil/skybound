//! The young of the blood-feeders: a ringed larva led by its sucker.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use core::f32::consts::PI;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A ringed larva.
pub const LARVA: u32 = 2;

/// A young blood-feeder: a ringed body led by its sucker, its fins growing out along its flanks.
pub fn leech_larva(p: Vec3, age: f32) -> Surface {
    let mut body = 1e6f32;
    for index in 0..7 {
        let along = index as f32 / 6.0;
        let radius = 0.20 + (along * PI).sin() * 0.13;
        body = smooth_min(
            body,
            ellipsoid(
                p - vec3(0.0, 0.0, (along - 0.5) * 1.35),
                vec3(radius, radius * 0.85, 0.19),
            ),
            0.06,
        );
    }
    let sucker = ellipsoid(p - vec3(0.0, 0.0, -0.67), vec3(0.25, 0.22, 0.12));
    body = smooth_min(body, sucker, 0.04);
    for pair in 0..2 {
        let fin = vec3(
            p.x.abs() - 0.28,
            p.y + 0.02,
            p.z - (pair as f32 - 0.5) * 0.5,
        );
        body = body.min(ellipsoid(fin, vec3(0.12 + age * 0.3, 0.035, 0.19)));
    }
    Surface::new(body, 0)
}
