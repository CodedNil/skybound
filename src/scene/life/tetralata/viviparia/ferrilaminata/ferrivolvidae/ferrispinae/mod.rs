use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, ellipsoid, eyes, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub mod custos;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ferrispinae", "Agile serpentine apex predators with razor spinal blades; juveniles often coordinate pack hunts.")
    .ecology("carnivore/predator", 6, false)
    .specimen("Shredders", 41.5);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 14.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.0, 1.2, 7.0), 0.7), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(15.0, 1.5, 8.0), 7.0), 1);
    hit.join(ellipsoid(p - vec3(0.0, 1.2, 0.5), vec3(0.2, 2.0, 5.6)), 1);
    hit.join(eyes(p, 7.0, 0.7), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.4, 0.45, 0.55, 0.05),
        vec3(0.65, 0.7, 0.8),
    )
}
