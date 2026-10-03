#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, ribbons, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(
    &super::TAXON,
    "Levita",
    "Smaller bubblebacks with light-sensitive trailing ribbons for subtle directional control.",
)
.ecology("herbivore", 13, false)
.specimen("Drifters", 25.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 5.0, 7.0, 5);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.5, 1.8, 5.0), 0.3), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(ribbons(p, c, 5), 4);
    hit.join(eyes(p, 5.0, 0.3), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.6, 0.8, 0.55, 0.04),
        vec3(0.6, 0.8, 0.4),
    )
}
