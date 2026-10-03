use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Luminiscidae", "Tiny intensely luminous algae-and-seed eaters in vast swarms. Important prey for small predators.")
    .ecology("herbivore", 15, false)
    .specimen("Sparkies", 19.8);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.4, 3.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.9, 0.8, 2.4), 0.65), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(5.0, 1.8, 2.0), 2.4), 1);
    hit.join(eyes(p, 2.4, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.9, 0.7, 0.2, 0.6),
        vec3(0.4, 0.6, 0.9),
    )
}
