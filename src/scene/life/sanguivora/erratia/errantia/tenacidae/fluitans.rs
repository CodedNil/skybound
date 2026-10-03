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
    "Fluitans",
    "Kelp-mat leeches with plant camouflage; attach permanently to passing grazers.",
)
.ecology("parasite", 5, false)
.specimen("Floating Ones", 26.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 9.0, 2);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.7, 0.65, 4.5), 0.45), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(ribbons(p, c, 2), 4);
    hit.join(eyes(p, 4.5, 0.45), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.4, 0.12, 0.24, 0.02),
        vec3(0.7, 0.25, 0.32),
    )
}
