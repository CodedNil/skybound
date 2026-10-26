use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, frond, gene, leaf, pick, rotate, speckle},
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, site},
    plantae::physiology::{BLADDER, FROND, Physiology, ROOT, STEM},
};
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.9,
    maturity_days: 365.0,
    lifespan_days: 40000.0,
    adult_mass: 25.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 25.0;

/// Kelp grows in forests, its holdfasts knotted together so the strands hang in one tangled jungle.
#[cfg(not(target_arch = "spirv"))]
pub const PHYSIOLOGY: Physiology = Physiology {
    stand: 24,
    ..super::PHYSIOLOGY
};

taxon!(
    Species,
    "Aerothallus vittatus",
    "Kelp",
    [
        "Kelp spreads long fibrous strands through the Gardens, with gas bladders supporting broad aur-collecting fronds at repeated intervals.",
        "Each strand bends across several neighbouring currents, distributing lift instead of hanging from a single buoyant crown.",
        "Dense growth interweaves into Jungles whose sheltered interior holds animals, attached plants and Weaver corridors.",
        "Grazers and Old Wise Ones browse the outer fronds, while broken tissue feeds smaller scavengers downstream.",
        "Spores establish new floating strands, allowing a forest to spread without a trunk rooted to any permanent surface.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(12.0, 8.0, 17.0).extend(0.0);
}

/// Where the strand runs at a distance `z` along it: weaving across the currents it spans, the
/// weave its own, swaying slowly as they shift.
fn strand(c: &CreatureInstance, z: f32) -> Vec3 {
    let weave = 0.5 + gene(c, 1.0) * 1.2;
    let sway = 1.0 + (c.animation.z * 0.3).sin() * 0.3;
    vec3(
        (z * (0.18 + gene(c, 2.0) * 0.2)).sin() * weave * sway,
        (z * (0.3 + gene(c, 3.0) * 0.2)).sin() * weave * 0.6,
        z,
    )
}

/// A frond of one of the forms kelp grows: a long, ruffled ribbon, a broad paddle, or a strap
/// split into fingers, `p` in the frond's own frame.
fn blade(p: Vec3, form: u32, scale: f32, ruffle: f32) -> f32 {
    match form {
        0 => frond(p, 6.0 * scale, 0.55, ruffle, 7.0),
        1 => leaf(p, 3.6 * scale, 1.5, ruffle * 0.6),
        _ => {
            let finger = (p.x / 0.32).round().clamp(-2.0, 2.0);
            let fan = vec3(p.x - finger * 0.32 * (p.z / 3.0).clamp(0.0, 1.0), p.y, p.z);
            leaf(fan, 4.2 * scale, 0.22, ruffle).min(leaf(p, 1.2 * scale, 0.8, 0.0))
        }
    }
}

/// A strand of kelp.
///
/// A long, fibrous stipe weaving through the currents it spans, with gas floats at its nodes and
/// fronds springing from each, and a clawed holdfast at its foot that knots it to the strands
/// around it into a forest. No two strands grow alike: each has its own length and weave, floats
/// round, podded or paired, fronds long and ruffled, broad, or split into fingers, and some fork
/// into side strands of their own.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let length = 9.0 + gene(c, 4.0) * 4.5;
    let thick = 0.14 + gene(c, 5.0) * 0.14;
    let along = p.z.clamp(-length, length);
    let mut shape = Surface::new((p - strand(c, along)).length() * 0.7 - thick, 0);
    let spacing = 2.2 + gene(c, 6.0) * 1.4;
    let float = pick(c, 7.0, 3);
    let form = pick(c, 8.0, 3);
    let fronds = count(c, 9.0, 1, 3);
    // Some strands' fronds stream up on the updraft, some hang heavy beneath the floats.
    let ruffle = -1.6 + gene(c, 10.0) * 2.6;
    let grown = 0.7 + gene(c, 15.0) * 0.8;
    // Only the nodes either side of the point can reach it.
    let near = (p.z / spacing).round();
    for step in -1..=1 {
        let node = near + step as f32;
        let z = node * spacing;
        if z.abs() > length - 0.5 {
            continue;
        }
        let root = strand(c, z);
        let size = 0.75 + speckle(vec2(node, 1.0), 1.0) * 0.5;
        shape.join(ellipsoid(p - root, vec3(0.34, 0.34, 0.42) * size), 0);
        let lift = p - root - vec3(0.0, 0.65, 0.0) * size;
        let bladder = match float {
            0 => ellipsoid(lift, vec3(0.45, 0.6, 0.45) * size),
            1 => ellipsoid(lift - vec3(0.0, 0.3, 0.0), vec3(0.3, 0.9, 0.3) * size),
            _ => {
                let twin = vec3(lift.x.abs() - 0.3 * size, lift.y, lift.z);
                ellipsoid(twin, vec3(0.32, 0.5, 0.32) * size)
            }
        };
        shape.join(bladder, 3);
        for i in 0..fronds {
            let hand = if (i + node.abs() as u32).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let angle = hand * (1.1 + i as f32 * 0.5) + speckle(vec2(node, i as f32), 2.0) * 0.4;
            let direction = vec3(angle.sin(), 0.05, angle.cos());
            let stalk = root + direction * 0.8;
            shape.join(capsule(p, root, stalk, 0.1), 1);
            let local = rotate(p - stalk, angle);
            let twist = (c.animation.z * 0.6 + z).sin() * 0.2;
            shape.join(blade(local, form, size * grown, ruffle + twist), 1);
        }
    }
    // A side strand forking from the main one and fronded at its tip.
    for branch in 0..count(c, 11.0, 0, 2) {
        let b = branch as f32;
        let from = strand(c, (b - 0.5) * length * 0.6);
        let side = if branch == 0 { 1.0 } else { -1.0 };
        let tip = from + vec3(side * (3.5 + gene(c, 12.0 + b) * 3.0), 1.0, 2.5);
        shape.join(tapered_tail(p, from, tip, thick * 0.9, thick * 0.5), 0);
        shape.join(
            ellipsoid(p - tip - vec3(0.0, 0.4, 0.0), vec3(0.35, 0.5, 0.35)),
            3,
        );
        shape.join(
            blade(rotate(p - tip, side * 1.2), form, 0.8 * grown, ruffle),
            1,
        );
    }
    // The holdfast's claws, which grip the next strand.
    let foot = strand(c, -length);
    for claw in 0..count(c, 13.0, 4, 7) {
        let angle = claw as f32 * 1.4 + gene(c, 14.0) * 3.0;
        let out = vec3(angle.cos(), angle.sin(), 0.0);
        let bend = foot + out * 0.9 - vec3(0.0, 0.0, 0.7);
        shape.join(tapered_tail(p, foot, bend, 0.16, 0.1), 0);
        shape.join(
            tapered_tail(p, bend, bend + out * 0.4 + vec3(0.0, 0.0, 0.6), 0.1, 0.03),
            0,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.00, 0.10), vec3(0.16, 0.16, 0.24)),
    site(BLADDER, vec3(0.00, 0.70, 0.00), vec3(0.20, 0.28, 0.20)),
    site(FROND, vec3(0.20, 0.02, 0.06), vec3(0.06, 0.06, 0.06)),
    site(ROOT, vec3(0.00, -0.18, -0.10), vec3(0.12, 0.10, 0.12)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.22, 0.46, 0.08), vec3(0.57, 0.69, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
