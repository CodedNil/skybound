use crate::scene::life::biology::Organs;
use crate::scene::life::biology::{Anatomy, Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    organs: Organs::ANIMAL,
    anatomy: Anatomy::Tetralate,
    wing_pairs: 2,
    oral_pairs: 1,
    feeding: Feeding::Grazer,
    speed: 12.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{CreatureInstance, tapered_tail};
use crate::scene::life::{Rank, Taxon};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub mod specialia;
pub mod vivipara;
pub mod volitantia;

pub const TAXON: Taxon = Taxon {
    name: "Tetralata",
    english_name: "Four Winged",
    description: "Tetralata have an elongated, limbless body with two ancestral pairs of wings and paired muscular mouth palps. \
        A continuous digestive crop passes food from the mouth to a posterior waste outlet. \
        Pleated air channels exchange respiratory gases, and an annular pulsatile transport network distributes dissolved nutrients. \
        The mouth palps, wing margins and tail specialise into feeding, attachment and manipulation structures throughout the phylum.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
};

pub fn oral_tentacles(
    point: Vec3,
    creature: &CreatureInstance,
    mouth: Vec3,
    scale: f32,
    pairs: u32,
) -> f32 {
    let bounds = ((point - mouth - vec3(0.0, -0.6, -0.4) * scale).abs()
        - vec3(1.3, 0.8, 0.8) * scale)
        .max(Vec3::ZERO)
        .length();
    if bounds > scale * 0.1 {
        return bounds * 0.7;
    }
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let mut distance = 1e6f32;
    for index in 0..pairs {
        let lane = index as f32;
        let root = mouth + vec3(0.14 + lane * 0.09, (lane % 2.0 - 0.5) * 0.16, lane * 0.10) * scale;
        let reach = scale * (1.5 - lane * 0.23);
        let sway = (creature.animation.z * 1.3 + lane * 1.9).sin() * 0.12;
        let control = root + vec3(0.12 + lane * 0.05, -0.8, -0.65) * reach;
        let tip = root + vec3(0.28 + sway + lane * 0.12, -0.5 + sway, -0.55) * reach;
        distance = distance.min(curved_tube(
            mirrored,
            root,
            control,
            tip,
            scale * (0.055 - lane * 0.006),
        ));
    }
    distance
}

fn curved_tube(point: Vec3, root: Vec3, control: Vec3, tip: Vec3, radius: f32) -> f32 {
    let linear = (control - root) * 2.0;
    let quadratic = tip - control * 2.0 + root;
    let mut fraction = ((point - root).dot(tip - root) / (tip - root).length_squared()).saturate();
    for _ in 0..4 {
        let tangent = linear + quadratic * (2.0 * fraction);
        let delta = root + linear * fraction + quadratic * fraction * fraction - point;
        let denominator = tangent.length_squared() + delta.dot(quadratic) * 2.0;
        fraction = (fraction - delta.dot(tangent) / denominator.abs().max(1e-5)).saturate();
    }
    let center = root + linear * fraction + quadratic * fraction * fraction;
    let taper = 1.0 - fraction * 0.88;
    let folds = 1.0 + (fraction * PI * 18.0).sin() * 0.04;
    ((point - center).length() - radius * taper * folds).min(tapered_tail(
        point,
        root,
        root + linear * 0.04,
        radius,
        radius * 0.92,
    )) * 0.7
}
