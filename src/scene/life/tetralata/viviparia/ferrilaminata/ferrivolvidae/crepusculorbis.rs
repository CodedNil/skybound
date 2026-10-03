use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, feathers, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Crepusculorbis", "Medium camouflaged mineral-field ambushers using luminous feathered-tail lures and blade-like claws; also scavenge.")
    .ecology("carnivore/predator", 2, false)
    .specimen("Duskslicers", 41.5);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 5.0, 14.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.2, 1.3, 5.0), 1.0), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(15.0, 1.5, 8.0), 5.0), 1);
    hit.join(feathers(p, c, 2.0), 1);
    hit.join(eyes(p, 5.0, 1.0), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.27, 0.2, 0.4, 0.08),
        vec3(0.4, 0.6, 0.9),
    )
}
