use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid, plantae,
    plantae::{rosette, rotate},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.43,
    maturity_days: 900.0,
    lifespan_days: 30000.0,
    adult_mass: 1.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 13.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Aurocorallum aureum",
    "Golden Coral",
    [
        "Golden Coral grows layered golden fans from branching attachments on large, slow-moving animals.",
        "Frilled surfaces collect dispersed aur while the host carries the colony through fresh air.",
        "The fans yield in a changing current, reducing strain on their attachments and avoiding the weight of a separate buoyant body.",
        "Moist recesses between branches retain nutrient-bearing dust and shelter small scavengers.",
        "The association remains most stable where modest growth leaves the host's breathing folds and lifting surfaces clear.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -2.0, 0.0), vec3(1.8, 0.6, 1.8)), 3);
    for i in 0..9 {
        let angle = i as f32 * 2.4;
        let start = vec3(angle.cos(), -2.0, angle.sin());
        let end = vec3(
            angle.cos() * 3.5,
            1.5 + (angle * 2.0).sin(),
            angle.sin() * 3.5,
        );
        shape.join(capsule(p, start, end, 0.22), 3);
        for j in 0..3 {
            let tip = end
                + vec3(
                    (angle + j as f32).cos() * 1.2,
                    j as f32 * 0.7,
                    (angle + j as f32).sin() * 1.2,
                );
            shape.join(capsule(p, end, tip, 0.12), 3);
            shape.join(rosette(rotate(p - tip, angle), 4, 1.0, 0.4, 0.5), 1);
        }
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.94, 0.06), vec3(0.43, 0.43, 0.43)),
    site(BLADDER, vec3(0.00, -1.69, -0.56), vec3(0.18, 0.18, 0.18)),
    pair(FROND, vec3(0.88, -1.88, 0.13), vec3(0.21, 0.21, 0.21)),
    site(ROOT, vec3(0.00, -1.69, 0.69), vec3(0.18, 0.18, 0.18)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.75, 0.42, 0.07), vec3(1.0, 0.7, 0.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
