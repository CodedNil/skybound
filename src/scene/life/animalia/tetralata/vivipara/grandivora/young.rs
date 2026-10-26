//! The young of the pouch-bearers, born long before they are formed to finish growing in the
//! pouch.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A curled newborn in the pouch.
pub const POUCH_YOUNG: u32 = 11;

/// Born long before it is formed, to finish growing in the pouch: a curled, translucent body, the
/// great dark eyes not yet open, the wings and mouth tentacles only nubs.
pub fn pouch_young(p: Vec3, age: f32) -> Surface {
    let grow = age / 0.2;
    let mut body = 1e6f32;
    for index in 0..7 {
        let along = index as f32 / 6.0;
        let bend = along * 2.6 - 0.6;
        let at = vec3(0.0, bend.sin() * 0.45 - 0.1, -bend.cos() * 0.45 + 0.15);
        let radius = 0.24 - along * 0.12 + (1.0 - along) * grow * 0.06;
        body = smooth_min(
            body,
            ellipsoid(p - at, vec3(radius, radius, radius * 0.9)),
            0.08,
        );
    }
    let side = vec3(p.x.abs(), p.y, p.z);
    let mut nubs = 1e6f32;
    for pair in 0..2 {
        nubs = nubs.min(ellipsoid(
            side - vec3(0.22, 0.12, -0.2 + pair as f32 * 0.3),
            vec3(0.1 + grow * 0.08, 0.025, 0.08),
        ));
    }
    let mut hit = Surface::new(smooth_min(body, nubs, 0.04), 0);
    hit.join(
        ellipsoid(side - vec3(0.14, -0.02, -0.58), vec3(0.09, 0.1, 0.09)),
        3,
    );
    hit
}
