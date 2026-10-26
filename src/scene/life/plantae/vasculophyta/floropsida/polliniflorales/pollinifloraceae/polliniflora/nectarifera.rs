use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
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
    plantae::vasculophyta::floropsida::BLOOM,
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 35.0,
    lifespan_days: 700.0,
    adult_mass: 0.5,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 11.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Polliniflora nectarifera",
    "Blooms",
    [
        "Blooms hold layered petals around reservoirs of sweet, aur-rich nectar on flexible stems.",
        "A buoyant body maintains the flower in rising air while the stem yields beneath an approaching Misting.",
        "Pollen brushes the visitor during feeding and travels onward when the animal leaves for another drifting plant.",
        "Hygroscopic surfaces supply water, and vascular channels deliver the minerals needed to renew petals and nectar.",
        "Their healing tissues are valued by Næ, so careful gathering preserves a living resource that also feeds the surrounding pollinator community.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 7.0, 6.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let stem = capsule(p, vec3(0.0, -1.5, 0.0), vec3(0.0, 0.3, 0.0), 0.18);
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -1.5, 0.0), vec3(1.2, 1.0, 1.2)), 3);
    shape.join(stem, 1);
    shape.join(rosette(p, 8, 4.3, 1.6, 1.4), 1);
    shape.join(
        rosette(rotate(p - vec3(0.0, 0.25, 0.0), 0.4), 6, 2.7, 1.1, 1.5),
        1,
    );
    shape.join(
        ellipsoid(p - vec3(0.0, 0.9, 0.0), vec3(0.85, 0.65, 0.85)),
        2,
    );
    for i in 0..7 {
        let angle = i as f32 * 0.897;
        let tip = vec3(angle.cos() * 0.6, 1.8, angle.sin() * 0.6);
        shape.join(capsule(p, vec3(0.0, 0.7, 0.0), tip, 0.04), 3);
        shape.join(ellipsoid(p - tip, Vec3::splat(0.13)), 2);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.94, -0.05), vec3(0.35, 0.35, 0.35)),
    site(BLADDER, vec3(0.00, 0.69, 0.05), vec3(0.35, 0.35, 0.35)),
    pair(FROND, vec3(0.54, -1.62, 0.11), vec3(0.42, 0.42, 0.42)),
    site(ROOT, vec3(0.00, -1.81, -0.48), vec3(0.39, 0.39, 0.39)),
    site(BLOOM, vec3(0.00, 1.06, -0.27), vec3(0.31, 0.31, 0.31)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.82, 0.16, 0.43), vec3(1.0, 0.68, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
