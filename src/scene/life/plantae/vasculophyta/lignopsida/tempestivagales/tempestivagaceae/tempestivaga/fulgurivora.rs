use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid, plantae,
    plantae::{leaf, rotate},
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
use core::f32::consts::FRAC_PI_2;
use spirv_std::glam::{Vec3, vec3};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1200.0,
    lifespan_days: 60000.0,
    adult_mass: 8.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 20.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Tempestivaga fulgurivora",
    "Stormchasers",
    [
        "Stormchasers carry broad upright sails around a slender woody body strengthened for turbulent air.",
        "Electrical changes in an approaching storm trigger adjustments that expose collecting veins while altering the plant's drift.",
        "Captured discharge passes into protected storage tissues, providing energy between charging opportunities.",
        "After a storm passes, changed sail orientation carries the plant onward rather than leaving it permanently in the most violent air.",
        "Their gathering is a warning of electrical activity, especially where humid geyser-fed currents meet stronger poleward winds.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(10.0, 12.0, 10.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -7.0, 0.0), vec3(0.0, 8.0, 0.0), 0.3),
        0,
    );
    shape.join(ellipsoid(p - vec3(0.0, -3.0, 0.0), vec3(1.3, 2.0, 1.3)), 3);
    for i in 0..4 {
        let local = rotate(p, i as f32 * FRAC_PI_2);
        let sail = vec3(local.x, local.z, local.y + 4.0);
        shape.join(leaf(sail, 11.0, 5.0, 1.4), 1);
        shape.join(
            capsule(local, vec3(0.0, 2.0, 0.0), vec3(4.5, 3.5, 0.0), 0.09),
            2,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.82, 0.09), vec3(0.42, 0.42, 0.42)),
    site(BLADDER, vec3(0.00, -2.46, -0.62), vec3(0.42, 0.42, 0.42)),
    pair(FROND, vec3(0.54, -3.21, -0.18), vec3(0.46, 0.46, 0.46)),
    site(ROOT, vec3(0.00, -4.18, 0.09), vec3(0.41, 0.41, 0.41)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.16, 0.22, 0.46), vec3(0.55, 0.7, 1.0));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        powered: |climate| climate.storm * 8.0,
        ..super::super::PHYSIOLOGY
    };
}
