pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{ellipsoid, smooth_min, tapered_tail};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
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
        "Members of Naeturus share four wings on short stalks, long tails trailing a train of forked streamers lit along their ribs by the aur sacs, feathery exchange plumes behind the head, and aur-storage organs isolated from the digestive circulation.",
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
    // The edge breaks into fine barbs rather than running clean, as a feather's does.
    let barbs = 1.0 - (along / length.max(1e-4) * 70.0).sin().abs() * 0.06;
    let breadth = width * (0.45 + 0.55 * (t * 3.0).min(1.0)) * (1.0 - t * t * 0.7) * barbs;
    let local = vec3(q.dot(across), q.dot(up), along - length * 0.5);
    ellipsoid(local, vec3(breadth, width * 0.1, length * 0.5)) * 0.8
}

/// A streamer of the genus's train.
///
/// A long, narrow vane from `root` along `direction` (of unit length), forked at its tip, with a
/// pale raised rib down its middle and three beads along the rib towards the tip, which the aur
/// sacs light from within with a trickle of their charge. Returns the vane, the beads and the rib.
///
/// The train is for one another: the genus greets and courts by spreading and rippling it, and in
/// the dim equatorial sky a herd's trains glow like strings of lamps, a sign of full, healthy sacs.
pub fn feather(point: Vec3, root: Vec3, direction: Vec3, length: f32, width: f32) -> Vec3 {
    let flat = vec3(direction.z, 0.0, -direction.x);
    let across = flat / flat.length().max(1e-4);
    let up = direction.cross(across);
    let q = point - root;
    let local = vec3(q.dot(across), q.dot(up), q.dot(direction));
    let notch = ellipsoid(
        local - vec3(0.0, 0.0, length * 0.98),
        vec3(width * 0.28, width, length * 0.14),
    );
    let blade = vane(point, root, direction, length, width).max(-notch);
    let rib = tapered_tail(
        point,
        root,
        root + direction * length * 0.9,
        width * 0.14,
        width * 0.06,
    );
    let mut beads = 1e6f32;
    for index in 0..3 {
        let at = vec3(0.0, 0.0, length * (0.55 + index as f32 * 0.14));
        beads = beads.min((local - at).length() - width * (0.17 - index as f32 * 0.02));
    }
    vec3(blade, beads, rib)
}

/// A feathery exchange plume of the genus.
///
/// A stem from `root` bending through `bend` to `tip`, `thickness` at its root, fringed on either
/// side with fine filaments `reach` long, `count` of them along it, which spread the air over the
/// most surface for the least weight.
pub fn plume(
    point: Vec3,
    root: Vec3,
    bend: Vec3,
    tip: Vec3,
    thickness: f32,
    reach: f32,
    count: f32,
) -> f32 {
    let stem = tapered_tail(point, root, bend, thickness, thickness * 0.7).min(tapered_tail(
        point,
        bend,
        tip,
        thickness * 0.7,
        thickness * 0.25,
    ));
    if stem > reach * 1.3 {
        return stem;
    }
    let first = (bend - root).length();
    let total = first + (tip - bend).length();
    let mut filaments = 1e6f32;
    for segment in 0..2 {
        let (a, b) = if segment == 0 {
            (root, bend)
        } else {
            (bend, tip)
        };
        let axis = b - a;
        let span = axis.length().max(1e-4);
        let direction = axis / span;
        let along = (point - a).dot(direction).clamp(0.0, span);
        // The nearest filament's root along the stem, and how far along the whole plume it is.
        let spacing = total / count;
        let at = ((along / spacing).round() * spacing).min(span);
        let overall = (if segment == 0 { 0.0 } else { first } + at) / total;
        let foot = a + direction * at;
        let flat = direction.cross(Vec3::Y);
        let side = flat / flat.length().max(1e-4);
        let sign = if (point - foot).dot(side) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let out = side * sign + direction * 0.7;
        let length = reach * (1.0 - overall * 0.65);
        let end = foot + out / out.length() * length;
        filaments = filaments.min(tapered_tail(
            point,
            foot,
            end,
            thickness * 0.3,
            thickness * 0.06,
        ));
    }
    smooth_min(stem, filaments * 0.85, thickness * 0.3)
}
