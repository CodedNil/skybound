//! The young of the porous gas-bags: a polyp, a soft fringed dome whose gas chambers swell as it
//! grows until it lifts away.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A polyp.
pub const POLYP: u32 = 1;

/// A porous animal's first form: a soft dome fringed at its rim, its gas chambers swelling as
/// it grows.
pub fn polyp(p: Vec3, age: f32, time: f32) -> Surface {
    let mut body = ellipsoid(p, vec3(0.7, 0.4 + age * 0.3, 0.75));
    let fringe = (vec2(vec2(p.x, p.z).length() - 0.64, p.y + 0.2).length()
        - 0.08
        - (p.x.atan2(p.z) * 24.0 + time * 3.0).sin() * 0.025)
        .max(p.y);
    body = smooth_min(body, fringe, 0.12);
    for index in 0..3 {
        let chamber = ellipsoid(
            p - vec3((index as f32 - 1.0) * 0.35, 0.25, 0.0),
            vec3(0.25, age * 0.6 + 0.05, 0.35),
        );
        body = smooth_min(body, chamber, 0.15);
    }
    Surface::new(body, 0)
}
