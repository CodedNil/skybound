use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, feathers, gills, ribbons, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Nætu", "Intelligent legless dragon-axolotl flyers with four elegant wings, external gills and variable-stiffness ventral ribbons. Aur sacs recharge in the mists and support healing and future telekinetic manipulation. Diverse sizes, shapes and colours. Næ is removed; Naetu is the playable species.")
    .ecology("omnivore", 1, true)
    .specimen("Naetu", 49.1);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.4, 15.0, 3);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.55, 1.65, 4.4), 1.0), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(19.0, 4.7, 6.0), 4.4), 1);
    hit.join(feathers(p, c, 2.7), 1);
    hit.join(gills(p, 4.4, 3), 2);
    hit.join(ribbons(p, c, 3), 4);
    hit.join(eyes(p, 4.4, 1.0), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.72, 0.83, 0.9, 0.08),
        vec3(0.43, 0.5, 0.78),
    )
}
