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
    "Aerogigans",
    "Enormous low-altitude bubblebacks, including forms adapted to humid climates.",
)
.ecology("herbivore", 3, false)
.specimen("Air Giants", 46.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 13.0, 12.0, 4);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(9.0, 11.0, 13.0), 0.4), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(ribbons(p, c, 4), 4);
    hit.join(eyes(p, 13.0, 0.4), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.35, 0.7, 0.45, 0.02),
        vec3(0.6, 0.8, 0.4),
    )
}
