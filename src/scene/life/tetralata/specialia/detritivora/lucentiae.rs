use crate::scene::life::tetralata::four_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, ribbons, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Lucentiae", "Long luminous filter-feeders that also lure and consume sparkies. Rise by day and descend toward the veilands at night.")
    .ecology("omnivore", 14, false)
    .specimen("Shimmerays", 40.4);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 17.0, 2);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.2, 1.0, 7.0), 0.55), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(four_wings(p, c, vec3(14.0, 5.0, 5.0), 7.0), 1);
    hit.join(ribbons(p, c, 2), 4);
    hit.join(eyes(p, 7.0, 0.55), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.3, 0.7, 0.85, 0.25),
        vec3(0.4, 0.6, 0.9),
    )
}
