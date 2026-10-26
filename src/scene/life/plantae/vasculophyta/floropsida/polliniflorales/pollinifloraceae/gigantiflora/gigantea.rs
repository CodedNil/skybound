use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid, plantae,
    plantae::{frond, rosette, rotate},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
    plantae::vasculophyta::floropsida::BLOOM,
};
use spirv_std::glam::{Vec3, vec3};

pub const BIOLOGY: Biology = Biology {
    size: 0.75,
    maturity_days: 400.0,
    lifespan_days: 18000.0,
    adult_mass: 12.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 30.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Gigantiflora gigantea",
    "Giant Blooms",
    [
        "Giant Blooms raise luminous flowers that approach the height of small trees above buoyant supporting bodies.",
        "Broad petals surround deep nectar reservoirs supplied through flexible vascular tissues.",
        "Their scale rewards large visitors whose energy needs cannot be met efficiently by ordinary Blooms.",
        "Slow, steady currents in fertile aur-rich Gardens support the heavy flowering structures and reduce damage during feeding.",
        "A rare successful visit can carry pollen across a large gap, making dependable attraction more useful than producing many tiny flowers.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(14.0, 20.0, 14.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -12.0, 0.0), vec3(0.0, 7.0, 0.0), 0.7),
        0,
    );
    shape.join(ellipsoid(p - vec3(0.0, -10.0, 0.0), vec3(3.5, 2.8, 3.5)), 3);
    shape.join(rosette(p - vec3(0.0, 7.0, 0.0), 9, 10.0, 3.0, 3.2), 1);
    shape.join(
        rosette(rotate(p - vec3(0.0, 8.0, 0.0), 0.35), 7, 6.5, 2.5, 4.0),
        1,
    );
    shape.join(ellipsoid(p - vec3(0.0, 8.5, 0.0), vec3(2.2, 1.2, 2.2)), 2);
    for i in 0..4 {
        shape.join(
            frond(
                rotate(p - vec3(0.0, -5.0 + i as f32 * 2.0, 0.0), i as f32 * 2.4),
                6.0,
                1.8,
                1.2,
                4.0,
            ),
            1,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -8.39, -0.38), vec3(0.95, 0.95, 0.95)),
    site(BLADDER, vec3(0.00, -9.46, 1.63), vec3(0.92, 0.92, 0.92)),
    pair(FROND, vec3(2.25, -9.64, -0.25), vec3(0.95, 0.95, 0.95)),
    site(ROOT, vec3(0.00, -10.89, -0.88), vec3(1.40, 1.40, 1.40)),
    site(BLOOM, vec3(0.00, -9.82, -2.38), vec3(0.91, 0.91, 0.91)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.48, 0.2, 0.77), vec3(0.28, 0.85, 1.0));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
