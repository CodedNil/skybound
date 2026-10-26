use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
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
    maturity_days: 5.0,
    lifespan_days: 1000.0,
    adult_mass: 0.05,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 9.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Aeromuscus velutinus",
    "Air Moss",
    [
        "Air Moss carpets bark and rough animal shells with dense green leaves that hold dew between their tips.",
        "The mat uses scattered aur light and dissolved nutrients retained from dust and decaying debris.",
        "Its low profile keeps most growth within the sheltered air close to a moving host.",
        "Tiny animals find moisture and refuge among the leaves, while slow decomposition returns their waste to the plant.",
        "Detached pieces can start another patch, making close passages between living communities important opportunities for dispersal.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 3.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -0.4, 0.0), vec3(3.5, 0.5, 3.5)), 1);
    for i in 0..23 {
        let angle = i as f32 * 2.39996;
        let radius = (i as f32 / 23.0).sqrt() * 3.2;
        let root = vec3(angle.cos() * radius, 0.0, angle.sin() * radius);
        let local = rotate(p - root, angle);
        shape.join(rosette(local, 5, 0.85, 0.16, 0.5), 1);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.29, 0.04), vec3(0.32, 0.15, 0.32)),
    site(BLADDER, vec3(0.00, -0.08, -0.40), vec3(0.14, 0.14, 0.14)),
    pair(FROND, vec3(2.05, -0.27, -0.09), vec3(0.14, 0.14, 0.14)),
    site(ROOT, vec3(0.00, -0.72, -0.04), vec3(0.14, 0.12, 0.14)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.12, 0.36, 0.07), vec3(0.45, 0.65, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
