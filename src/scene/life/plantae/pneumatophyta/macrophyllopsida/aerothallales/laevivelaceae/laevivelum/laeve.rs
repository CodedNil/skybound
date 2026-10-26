use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
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
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.1,
    maturity_days: 730.0,
    lifespan_days: 60000.0,
    adult_mass: 18.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 17.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Laevivelum laeve",
    "Smooth Sails",
    [
        "Smooth Sails spread broad, thin aur-collecting membranes almost horizontally through rising air.",
        "Gentle undulation changes drag and sheds uneven loads of rain before they distort the whole surface.",
        "Small marginal glands produce a soft glow where the surrounding light is dim.",
        "Overlapping sails cool and shelter the air below, creating patches useful to moisture-sensitive organisms.",
        "Windborne spores establish new membranes, while browsing Old Wise Ones keep mature sails connected to the wider food web.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(11.0, 3.0, 11.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let radius = p.xz().length();
    let height =
        (radius * 0.65).sin() * 0.25 + (p.x * 0.7 + c.animation.z * 0.4).sin() * radius * 0.035;
    let mut shape = Surface::new(
        (p.y - height).abs().max((radius - 9.0) * 0.5) * 0.65 - 0.045,
        1,
    );
    for i in 0..20 {
        let angle = i as f32 * TAU / 20.0;
        shape.join(
            ellipsoid(
                p - vec3(
                    angle.cos() * 8.9,
                    (8.9f32 * 0.65).sin() * 0.25,
                    angle.sin() * 8.9,
                ),
                Vec3::splat(0.14),
            ),
            2,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.03, -0.10), vec3(0.03, 0.03, 0.03)),
    site(BLADDER, vec3(0.00, 0.03, 0.10), vec3(0.03, 0.03, 0.03)),
    pair(FROND, vec3(4.52, 0.05, -0.20), vec3(0.03, 0.03, 0.03)),
    site(ROOT, vec3(0.00, 0.08, -0.49), vec3(0.04, 0.04, 0.04)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.17, 0.5, 0.36), vec3(0.29, 0.95, 0.7));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
