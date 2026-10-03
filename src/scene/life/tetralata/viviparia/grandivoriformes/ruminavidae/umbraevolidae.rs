use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, feathers, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Umbraevolidae", "Large intelligent scavengers with light-absorbing dark plumage, sensitive thermal perception and coordinating warning cries.")
    .ecology("carnivore/scavenger", 3, false)
    .specimen("Shadowings", 49.8);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 8.0, 14.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(4.0, 4.0, 8.0), 1.2), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(18.0, 6.0, 7.0), 8.0), 1);
    hit.join(feathers(p, c, 2.5), 1);
    hit.join(eyes(p, 8.0, 1.2), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.17, 0.21, 0.32, 0.04),
        vec3(0.4, 0.6, 0.9),
    )
}
