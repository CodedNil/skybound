//! A plant's spore under its sail. What takes hold grows as a small plant of its parent's shape,
//! drawn by the grown plant's own tracer.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A spore under its sail.
pub const SPORE: u32 = 5;

/// A plant's spore, under a sail that the updraft carries.
pub fn spore(p: Vec3) -> Surface {
    let body = ellipsoid(p, vec3(0.3, 0.45, 0.3));
    let sail = ellipsoid(p - vec3(0.0, 0.4, 0.0), vec3(0.7, 0.06, 0.65));
    Surface::new(smooth_min(body, sail, 0.06), 2)
}
