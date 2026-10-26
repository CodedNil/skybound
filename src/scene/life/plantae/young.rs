//! The young of plants: a spore under its sail, a seedling, or a plant dividing in two.

use crate::scene::life::{ellipsoid, plantae, smooth_min, surface::Surface};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A spore under its sail.
pub const SPORE: u32 = 5;

/// A seedling.
pub const SPROUT: u32 = 6;

/// A plant dividing in two.
pub const BUD: u32 = 7;

/// A plant's spore, under a sail that the updraft carries.
pub fn spore(p: Vec3) -> Surface {
    let body = ellipsoid(p, vec3(0.3, 0.45, 0.3));
    let sail = ellipsoid(p - vec3(0.0, 0.4, 0.0), vec3(0.7, 0.06, 0.65));
    Surface::new(smooth_min(body, sail, 0.06), 2)
}

/// A plant's seedling: a bulb on a stalk, opening its first leaves.
pub fn sprout(p: Vec3, age: f32, time: f32) -> Surface {
    let bulb = ellipsoid(p, vec3(0.35, 0.3, 0.35));
    let stalk = ellipsoid(p - vec3(0.0, 0.3, 0.0), vec3(0.06, 0.5, 0.06));
    let mut body = smooth_min(bulb, stalk, 0.06);
    for index in 0..4 {
        let angle = index as f32 * TAU / 4.0;
        let turned = plantae::rotate(p - vec3(0.0, 0.1 + index as f32 * 0.12, 0.0), angle);
        body = body.min(plantae::leaf(
            turned,
            0.35 + age * 0.75,
            0.14 + age * 0.22,
            0.08 + (time + angle).sin() * 0.04,
        ));
    }
    Surface::new(body, 3)
}

/// A plant dividing into two.
pub fn bud(p: Vec3, age: f32) -> Surface {
    let body = smooth_min(
        ellipsoid(p - vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
        ellipsoid(p + vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
        0.15,
    );
    Surface::new(body, 3)
}
