use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, legs, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ramiphagae", "Slow armored grazers with crushing mandibles that strip Brambles carefully and recycle woody nutrients.")
    .ecology("herbivore", 2, false)
    .specimen("Driftmunchers", 26.8);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.5, 3.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.5, 1.5, 3.5), 0.7), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.distance = hit.distance.min(legs(p, vec3(2.5, 1.5, 3.5), 3));
    hit.join(four_wings(p, c, vec3(8.0, 3.0, 3.0), 3.5), 1);
    hit.join(eyes(p, 3.5, 0.7), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.45, 0.55, 0.3, 0.0),
        vec3(0.4, 0.6, 0.9),
    )
}
