use crate::scene::life::tetralata::wing;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Lucinani", "Tiny luminous mutualists living inside Whisperwood. Sap pheromones coordinate their flight to move the trees. Descended from parasitic mimics.")
    .ecology("herbivore", 1, false)
    .specimen("Glowgnomes", 16.0);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 1.5, 5.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.8, 2.0, 1.5), 0.65), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(
        wing(
            p,
            vec3(3.0, 0.0, 1.5),
            vec3(c.animation.x, c.animation.y, 1.0),
        ),
        1,
    );
    hit.join(eyes(p, 1.5, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.95, 0.65, 0.2, 0.65),
        vec3(0.7, 0.25, 0.32),
    )
}
