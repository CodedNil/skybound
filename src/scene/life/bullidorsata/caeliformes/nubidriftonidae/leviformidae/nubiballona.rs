#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, ribbons, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Nubiballona", "High-altitude bubblebacks with smooth reflective skin that breaks up their outline to predators.")
    .ecology("herbivore", 4, false)
    .specimen("Cloud Balloons", 15.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.0, 0.0, 3);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(3.0, 4.0, 3.0), 0.2), 0);
    hit.join(ribbons(p, c, 3), 4);
    hit.join(eyes(p, 3.0, 0.2), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.35, 0.7, 0.45, 0.02),
        vec3(0.6, 0.8, 0.4),
    )
}
