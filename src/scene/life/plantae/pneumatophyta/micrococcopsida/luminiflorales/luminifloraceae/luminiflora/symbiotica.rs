use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Function, Organ, Viscera, pair, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.19,
    maturity_days: 2.0,
    lifespan_days: 120.0,
    adult_mass: 0.03,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 7.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Luminiflora symbiotica",
    "Lumaweed",
    [
        "Lumaweed grows as rapidly expanding luminous patches on kelp, animal coverings and other living surfaces.",
        "It collects dispersed aur and releases a soft glow in dim skies, making occupied branches visible in mineral haze.",
        "Fragments readily establish new patches when neighbouring organisms brush together.",
        "Animal hosts receive some of its nutrients, but plant hosts lose resources and collecting space to the same growth.",
        "Small grazers harvest the edible tissue, limiting its spread and carrying production from a host-bound patch into the surrounding food web.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 3.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -0.4, 0.0), vec3(2.8, 0.3, 2.8)), 1);
    for i in 0..15 {
        let angle = i as f32 * 2.39996;
        let radius = (i as f32 / 15.0).sqrt() * 2.4;
        let root = vec3(angle.cos() * radius, 0.0, angle.sin() * radius);
        let tip = root
            + vec3(
                0.18 * angle.sin(),
                0.5 + (angle * 3.0).sin().abs(),
                0.2 * angle.cos(),
            );
        shape.join(capsule(p, root, tip, 0.07), 1);
        shape.join(ellipsoid(p - tip, vec3(0.18, 0.28, 0.18)), 2);
    }
    shape
}

/// The glow gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const GLOW: Organ = Organ::new("glow gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.51, 0.18), vec3(0.13, 0.13, 0.13)),
    site(BLADDER, vec3(0.00, -0.35, -0.04), vec3(0.20, 0.20, 0.20)),
    pair(FROND, vec3(1.07, -0.27, -0.07), vec3(0.10, 0.10, 0.10)),
    site(ROOT, vec3(0.00, -0.40, 0.39), vec3(0.21, 0.21, 0.21)),
    site(GLOW, vec3(0.00, -0.24, -0.39), vec3(0.11, 0.11, 0.11)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.09, 0.4, 0.22), vec3(0.15, 0.95, 0.8));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation, epiphyte};

    pub const PHYSIOLOGY: Physiology = Physiology {
        feeds_host: true,
        propagation: Propagation::Fragmentation,
        attaches: true,
        partner: epiphyte,
        ..super::super::PHYSIOLOGY
    };
}
