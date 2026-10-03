#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(
    &super::TAXON,
    "Hematozoa",
    "Bloodstream parasites whose external juveniles detach to seek new hosts.",
)
.ecology("parasite", 53, false)
.specimen("Blood Dwellers", 17.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.0, 5.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.5, 0.5, 2.0), 0.3), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(eyes(p, 2.0, 0.3), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.4, 0.12, 0.24, 0.02),
        vec3(0.4, 0.12, 0.24),
    )
}
