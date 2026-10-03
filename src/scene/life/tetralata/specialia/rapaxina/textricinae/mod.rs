use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, legs, ribbons, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub mod sputorexidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Textricinae", "Silk-spinning aerial engineers that connect plants into fields and trap prey. Some build nests and collect shiny objects.")
    .ecology("carnivore/predator", 142, false)
    .specimen("Weavers", 36.7);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.0, 4.0, 3);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.5, 1.0, 3.0), 0.8), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.distance = hit.distance.min(legs(p, vec3(1.5, 1.0, 3.0), 2));
    hit.join(four_wings(p, c, vec3(12.0, 3.5, 6.0), 3.0), 1);
    hit.join(ribbons(p, c, 3), 4);
    hit.join(eyes(p, 3.0, 0.8), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.65, 0.45, 0.75, 0.08),
        vec3(0.4, 0.6, 0.9),
    )
}
