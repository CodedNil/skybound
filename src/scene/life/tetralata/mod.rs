#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
use spirv_std::glam::{FloatExt, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub mod specialia;
pub mod viviparia;
pub mod volitilia;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Tetralata", "").ecology("", 0, false);

pub fn wing(p: Vec3, dimensions: Vec3, pose: Vec3) -> f32 {
    let (span, root, chord) = (dimensions.x, dimensions.y, dimensions.z);
    let (stroke, tuck, sweep) = (pose.x, pose.y, pose.z);
    let x = p.x.abs();
    let reach = span * (1.0 - tuck * 0.65);
    let u = ((x - 1.0) / reach).saturate();
    let center = root + u * u * sweep + tuck * u * 7.0;
    let height = 0.7
        + stroke * x * (1.0 - tuck * 0.7)
        + u * u * 2.0
        + (x * 0.7 + p.z * 0.5).sin() * u * 0.25;
    // A swept elliptical membrane; the distance safety factor covers its bend.
    let edge = ((x - 1.0 - reach * 0.5) / (reach * 0.5)).powi(2) + ((p.z - center) / chord).powi(2)
        - 1.0
        + (u * 42.0).sin() * u * u * 0.09;
    (edge * 0.65).max((p.y - height).abs() - 0.18) * 0.45
}

pub fn four_wings(p: Vec3, c: &super::CreatureInstance, dimensions: Vec3, length: f32) -> f32 {
    wing(
        p,
        vec3(dimensions.x, 0.0, dimensions.y),
        vec3(c.animation.x, c.animation.y, dimensions.z),
    )
    .min(wing(
        p,
        vec3(dimensions.x * 0.7, length * 1.4, dimensions.y * 0.75),
        vec3(c.animation.x * 0.75, c.animation.y, dimensions.z),
    ))
}
