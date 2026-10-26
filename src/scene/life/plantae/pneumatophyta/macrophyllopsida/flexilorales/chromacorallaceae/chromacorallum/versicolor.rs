use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid, plantae,
    plantae::{rosette, rotate},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.25,
    maturity_days: 180.0,
    lifespan_days: 9000.0,
    adult_mass: 0.2,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 10.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Chromacorallum versicolor",
    "Color Coral",
    [
        "Colour Coral spreads branching fans of green, yellow, pink, blue and purple across kelp and Laevensail growth.",
        "Frilled surfaces collect dispersed aur while narrow attachments leave much of the host exposed to passing air.",
        "Flexible junctions let each fan turn independently when a long frond twists in the current.",
        "A faint glow reveals living margins in the dim equatorial skies and attracts small organisms into the sheltered branches.",
        "The colony gains support and transport, while excessive growth can shade or weigh down the plant carrying it.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 6.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -2.0, 0.0), vec3(0.0, 1.0, 0.0), 0.32),
        0,
    );
    for i in 0..7 {
        let angle = i as f32 * 2.4;
        let root = vec3(0.0, -1.2 + i as f32 * 0.45, 0.0);
        let tip = root + vec3(angle.cos() * 2.3, 1.8, angle.sin() * 2.3);
        shape.join(capsule(p, root, tip, 0.14), 3);
        let local = rotate(p - tip, angle);
        shape.join(rosette(local, 5, 2.1, 0.8, 0.5), 1);
        shape.join(ellipsoid(p - tip, Vec3::splat(0.18)), 2);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.59, 0.04), vec3(0.24, 0.24, 0.24)),
    site(BLADDER, vec3(0.00, 1.66, 0.22), vec3(0.10, 0.10, 0.10)),
    site(FROND, vec3(1.16, 1.61, 1.52), vec3(0.10, 0.10, 0.10)),
    site(ROOT, vec3(0.00, -1.34, 0.04), vec3(0.23, 0.23, 0.23)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.42, 0.11, 0.55), vec3(0.1, 0.8, 0.95));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
