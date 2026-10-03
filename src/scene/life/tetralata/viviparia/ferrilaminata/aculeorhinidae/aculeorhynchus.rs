use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, ellipsoid, eyes, smooth_min, tail, tapered_tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(
    &super::TAXON,
    "Aculeorhynchus",
    "Diving predators with a huge needle-like snout aimed at vital organs.",
)
.ecology("carnivore/predator", 2, false)
.specimen("Needlesharks", 41.5);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 6.0, 16.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.2, 1.5, 6.0), 0.65), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    let root = vec3(0.0, 0.325, -7.3);
    hit.distance = hit
        .distance
        .min(tapered_tail(p, root, root - Vec3::Z * 5.0, 0.5, 0.02));
    hit.join(four_wings(p, c, vec3(15.0, 1.5, 8.0), 6.0), 1);
    hit.join(ellipsoid(p - vec3(0.0, 1.5, 0.5), vec3(0.2, 1.0, 4.8)), 1);
    hit.join(eyes(p, 6.0, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.4, 0.45, 0.55, 0.05),
        vec3(0.4, 0.7, 0.8),
    )
}
