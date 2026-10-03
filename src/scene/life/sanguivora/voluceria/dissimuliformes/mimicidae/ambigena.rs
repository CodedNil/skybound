use crate::scene::life::tetralata::wing;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, legs, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ambigena", "Mimic grooming Sweepers with specialized limbs, cleaning antiseptically while siphoning fluids.")
    .ecology("parasite", 6, false)
    .specimen("Double Natured", 22.6);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.5, 2.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.4, 2.0, 2.5), 0.7), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.distance = hit.distance.min(legs(p, vec3(2.4, 2.0, 2.5), 2));
    hit.join(
        wing(
            p,
            vec3(6.0, 0.0, 3.0),
            vec3(c.animation.x, c.animation.y, 2.0),
        ),
        1,
    );
    hit.join(eyes(p, 2.5, 0.7), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.65, 0.6, 0.32, 0.1),
        vec3(0.7, 0.25, 0.32),
    )
}
