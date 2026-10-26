pub use super::{BIOLOGY, render};
use crate::scene::life::{
    capsule, ellipsoid,
    plantae::{frond, leaf, rotate},
    surface::Surface,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub mod gigantiflora;
pub mod polliniflora;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Pollinifloraceae",
    "Pollen Flowers",
    [
        "Members of this family use nectar-filled flowers to recruit animals as pollen carriers.",
        "Layered petals and luminous tissues present signals that remain useful through changing aur brightness.",
        "Buoyant bodies and flexible vascular stems support the weight of the flower and temporary visitors.",
        "Small forms reward frequent visits, while giant forms supply enough nectar for larger travellers.",
        "Stored aur also gives some tissues healing value to Næ, connecting reproduction to careful gathering as well as animal feeding.",
    ],
);

/// How a flower is built: its petals, their form and how far they open, and what lies within.
#[derive(Clone, Copy)]
pub struct Flower {
    /// Petals in each whorl.
    pub petals: u32,
    /// Whorls of petals, each inside and above the last.
    pub whorls: u32,
    /// Pointed, rounded, frilled, or long and trailing.
    pub form: u32,
    pub length: f32,
    pub width: f32,
    /// How far the petals rise from level, in radians: below zero they hang as a bell.
    pub open: f32,
    /// Stamens standing in the throat, their tips glowing with pollen.
    pub stamens: u32,
}

/// A flower, `p` in its own frame with its throat facing up the y axis: whorls of petals round a
/// glowing cup of nectar, with stamens standing in it that dust each visitor with pollen.
pub fn flower(p: Vec3, flower: Flower, time: f32) -> Surface {
    let Flower {
        petals,
        whorls,
        form,
        length,
        width,
        open,
        stamens,
    } = flower;
    let mut shape = Surface::new(1e6, 1);
    for whorl in 0..whorls {
        let w = whorl as f32;
        let scale = 1.0 - w * 0.28;
        let rise = open + w * 0.35;
        let (sine, cosine) = rise.sin_cos();
        for i in 0..petals {
            let angle = (i as f32 + w * 0.5) * TAU / petals as f32;
            let q = rotate(p - vec3(0.0, w * length * 0.08, 0.0), angle);
            // Tilted up out of level by how far the petal has opened.
            let q = vec3(q.x, q.y * cosine - q.z * sine, q.z * cosine + q.y * sine);
            let along = length * scale;
            let petal = match form {
                0 => leaf(q, along, width * scale, along * 0.08),
                1 => leaf(q, along * 0.8, width * scale * 1.5, along * 0.12),
                2 => frond(q, along, width * scale * 1.2, along * 0.1, 3.0),
                _ => {
                    let droop = (time * 0.6 + angle).sin() * 0.1;
                    leaf(q, along * 1.5, width * scale * 0.45, -along * (0.3 + droop))
                }
            };
            shape.join(petal, 1);
        }
    }
    shape.join(
        ellipsoid(
            p - vec3(0.0, length * 0.05, 0.0),
            vec3(1.0, 0.55, 1.0) * length * 0.22,
        ),
        2,
    );
    for i in 0..stamens {
        let angle = i as f32 * TAU / stamens as f32;
        let tip = vec3(
            angle.cos() * length * 0.15,
            length * 0.35,
            angle.sin() * length * 0.15,
        );
        shape.join(capsule(p, Vec3::ZERO, tip, length * 0.012), 3);
        shape.join((p - tip).length() - length * 0.035, 2);
    }
    shape
}
