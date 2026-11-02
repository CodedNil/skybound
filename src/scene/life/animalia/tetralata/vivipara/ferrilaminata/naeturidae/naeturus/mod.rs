pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{ellipsoid, smooth_min, tapered_tail};
use spirv_std::glam::{Vec3, vec3};
pub mod nae;
pub mod naetu;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Naeturus",
    "Nae Kin",
    [
        "Members of Naeturus share four wings on short stalks, long tails flighted with feather vanes, and aur-storage organs isolated from the digestive circulation.",
        "The genus has diverged into small dexterous Næ and enormous carrying Nætu.",
        "Mouth tentacles retain their ancestral feeding role while also supporting grooming, construction and close social contact.",
        "Their complementary sizes let one lineage maintain dwellings and tools while the other carries families between distant communities.",
        "Shared recognition and dependence on safely stored aur sustain the association through journeys that either form would struggle to make alone.",
    ],
);

/// A tail of the genus: a shaft tapering from `start` to `end` thick along the joints `joint`
/// gives, `count` of them, each running smoothly into the next so it bends as one.
pub fn shaft<F: Fn(u32) -> Vec3>(point: Vec3, joint: F, count: u32, start: f32, end: f32) -> f32 {
    let mut distance = 1e6f32;
    for index in 0..count - 1 {
        let from = index as f32 / (count - 1) as f32;
        let to = (index + 1) as f32 / (count - 1) as f32;
        let a = start + (end - start) * from;
        let b = start + (end - start) * to;
        distance = smooth_min(
            distance,
            tapered_tail(point, joint(index), joint(index + 1), a, b),
            a * 0.5,
        );
    }
    distance
}

/// One feather vane of a tail: a long, flat blade from `root` along `direction` (of unit length),
/// lying flat across the air as the tail does, widest a third of the way out and narrowing to a
/// rounded tip.
pub fn vane(point: Vec3, root: Vec3, direction: Vec3, length: f32, width: f32) -> f32 {
    let flat = vec3(direction.z, 0.0, -direction.x);
    let across = flat / flat.length().max(1e-4);
    let up = direction.cross(across);
    let q = point - root;
    let along = q.dot(direction);
    let t = (along / length).clamp(0.0, 1.0);
    let breadth = width * (0.45 + 0.55 * (t * 3.0).min(1.0)) * (1.0 - t * t * 0.6);
    let local = vec3(q.dot(across), q.dot(up), along - length * 0.5);
    ellipsoid(local, vec3(breadth, width * 0.1, length * 0.5)) * 0.8
}
