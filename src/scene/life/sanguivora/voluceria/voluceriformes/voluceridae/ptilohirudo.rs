use crate::scene::life::tetralata::wing;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Taxon, prepare_pose};
use crate::scene::{
    life::{CreatureInstance, eyes, feathers, gills, smooth_min, tail, torso},
    surface::{Hit, Ray, Surface, trace},
};
use spirv_std::glam::{Vec3, vec3, vec4};

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ptilohirudo", "Small feather-winged hit-and-run gliders with vision adapted to low aur light and dense clouds.")
    .ecology("parasite", 2, false)
    .specimen("Feathered Leeches", 30.4);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 5.0, 0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.7, 0.65, 4.5), 0.45), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(
        wing(
            p,
            vec3(9.0, 0.0, 3.0),
            vec3(c.animation.x, c.animation.y, 5.0),
        ),
        1,
    );
    hit.join(feathers(p, c, 1.1), 1);
    hit.join(gills(p, 4.5, 3), 2);
    hit.join(eyes(p, 4.5, 0.45), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.4, 0.12, 0.24, 0.02),
        vec3(0.7, 0.25, 0.32),
    )
}
