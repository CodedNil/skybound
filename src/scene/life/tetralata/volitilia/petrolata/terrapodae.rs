use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, legs, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Terrapodae", "Rare massive mineral-crusted animals slowly sinking through lower mid-levels, feeding on Root-Stars and mineral geysers.")
    .ecology("herbivore", 0, false)
    .specimen("Grounders", 46.8);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 9.0, 3.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(7.0, 5.0, 9.0), 0.8), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.distance = hit.distance.min(legs(p, vec3(7.0, 5.0, 9.0), 3));
    hit.join(four_wings(p, c, vec3(18.0, 6.0, 4.0), 9.0), 1);
    hit.join(eyes(p, 9.0, 0.8), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.35, 0.33, 0.3, 0.0),
        vec3(0.4, 0.6, 0.9),
    )
}
