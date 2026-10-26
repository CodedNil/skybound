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
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.36,
    maturity_days: 7300.0,
    lifespan_days: 365_000.0,
    adult_mass: 60.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 32.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Susurrus tubulatus",
    "Whisperwood",
    [
        "Whisperwood rises around a broad-based hollow trunk whose ridged walls guide the updraft through the body.",
        "Tubular branches end in apertures that whistle at different pitches, surrounding delicate fan leaves with an audible record of the passing air.",
        "Glowgnomes occupy sheltered wood cavities and receive sap while providing light and collective pulling force.",
        "Six vertical sap veins carry pheromone requests to groups on different sides, allowing gradual movement toward favourable humidity or away from danger.",
        "The partnership preserves lift and living support without roots, making an old tree a navigable refuge that travels with its inhabitants.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(13.0, 22.0, 13.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let radial = p.xz().length();
    let radius = 1.8 - p.y * 0.04;
    let rings = (p.y * 2.5).cos().smoothstep(0.75, 0.95) * 0.15;
    let trunk = ((radial - radius - rings).abs() - 0.22).max(p.y.abs() - 17.0) * 0.75;
    let mut shape = Surface::new(trunk, 0);
    for i in 0..12 {
        let angle = i as f32 * 2.4;
        let start = vec3(
            angle.cos() * radius,
            i as f32 * 2.1 - 9.0,
            angle.sin() * radius,
        );
        let end = start + vec3(angle.cos() * 6.0, 2.0, angle.sin() * 6.0);
        let branch = capsule(p, start, end, 0.25).max(-capsule(p, start, end, 0.14));
        shape.join(branch, 0);
        let mut local = rotate(p - end, angle);
        local.z += 1.0;
        shape.join(rosette(local, 5, 3.6, 1.3, 0.6), 1);
        shape.join(ellipsoid(p - start, Vec3::splat(0.14)), 2);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.59, -1.74), vec3(0.12, 0.12, 0.12)),
    site(BLADDER, vec3(0.00, 7.27, -1.74), vec3(0.08, 0.08, 0.08)),
    pair(FROND, vec3(1.62, 1.18, -0.70), vec3(0.13, 0.13, 0.13)),
    site(ROOT, vec3(0.00, -10.02, -2.67), vec3(0.13, 0.13, 0.13)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.15, 0.44, 0.3), vec3(0.37, 0.85, 0.56));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
