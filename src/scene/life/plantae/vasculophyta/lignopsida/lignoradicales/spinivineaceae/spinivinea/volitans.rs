use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule,
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
    maturity_days: 1000.0,
    lifespan_days: 70000.0,
    adult_mass: 8.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 13.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Spinivinea volitans",
    "Brambles",
    [
        "Brambles form dense spheres and linked chains of woody vines supported by large lotus-shaped leaves.",
        "Their thorns carry a mild auric charge that discourages most large grazers from pressing into the tangle.",
        "Small animals use interior gaps as shelter, while Driftmunchers work carefully through bark and spines on the outside.",
        "Flexible connections let a chain bend around uneven airflow rather than behave as a rigid floating wall.",
        "Spores establish new tangles as older chains separate, continually changing the passages and refuges inside a Bramble community.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(1e6, 0);
    for i in 0..6 {
        let local = rotate(p, i as f32 * 0.52);
        let radial = vec3(local.x, local.y * (0.9 + i as f32 * 0.03), 0.0).length();
        shape.join(vec3(radial - 3.5, 0.0, local.z).length() - 0.2, 0);
        for j in 0..6 {
            let angle = j as f32 * 1.047 + i as f32;
            let start = vec3(angle.cos() * 3.5, angle.sin() * 3.5, 0.0);
            let tip = start * 1.32;
            shape.join(capsule(local, start, tip, 0.09), 2);
        }
    }
    shape.join(rosette(p - vec3(0.0, 2.9, 0.0), 7, 5.0, 1.4, 0.8), 1);
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.56, -3.44), vec3(0.13, 0.13, 0.13)),
    site(BLADDER, vec3(0.00, 3.44, -0.81), vec3(0.15, 0.15, 0.15)),
    pair(FROND, vec3(2.88, 1.12, -1.62), vec3(0.13, 0.13, 0.13)),
    site(ROOT, vec3(0.00, -3.44, -0.81), vec3(0.15, 0.15, 0.15)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.16, 0.35, 0.1), vec3(0.4, 0.6, 0.14));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
