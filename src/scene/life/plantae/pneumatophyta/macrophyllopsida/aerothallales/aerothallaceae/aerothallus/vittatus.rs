use crate::scene::life::{
    Look, Plant,
    biology::Biology,
    ellipsoid,
    plantae::{count, gene, speckle},
    smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, site},
    plantae::physiology::{BLADDER, FROND, Physiology, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.9,
    maturity_days: 365.0,
    lifespan_days: 40000.0,
    adult_mass: 25.0,
    altitude: (2000.0, 7000.0),
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
        "Kelp spreads long fibrous strands through the Gardens, wound with a thin spiral vane and hung at repeated nodes with chains of glassy gas beads above fans of glowing, charged hairs that gather aur.",
        "Each strand bends across several neighbouring currents, distributing lift instead of hanging from a single buoyant crown.",
        "Dense growth interweaves into Jungles whose sheltered interior holds animals, attached plants and Weaver corridors.",
        "Grazers and Old Wise Ones browse the outer vanes and fans, while broken tissue feeds smaller scavengers downstream.",
        "Spores establish new floating strands, allowing a forest to spread without a trunk rooted to any permanent surface.",
    ],
);

pub fn animate(c: &mut Plant) {
    c.opaque_bounds = vec3(12.0, 8.0, 17.0).extend(0.0);
}

/// Where the strand runs at a distance `z` along it: weaving across the currents it spans, the
/// weave its own, swaying slowly as they shift.
fn strand(c: &Plant, z: f32) -> Vec3 {
    let weave = 0.5 + gene(c, 1.0) * 1.2;
    let sway = 1.0 + (c.animation.z * 0.3).sin() * 0.3;
    vec3(
        (z * (0.18 + gene(c, 2.0) * 0.2)).sin() * weave * sway,
        (z * (0.3 + gene(c, 3.0) * 0.2)).sin() * weave * 0.6,
        z,
    )
}

/// A fan of collecting filaments at a node, `p` from the node: stiff, fine hairs splayed down and
/// out towards the aur below, each charged so the dispersed aur drifting up through the forest
/// settles on it, and each tipped with a bead glowing with what it has gathered. Returns the hairs
/// and the beads apart.
fn fan(p: Vec3, hairs: u32, reach: f32, spread: f32, turn: f32) -> (f32, f32) {
    // Far from the fan, the distance to a sphere round it is enough; it is never small enough to
    // count as touching, so the sphere itself is never drawn.
    let bound = p.length() - reach * 1.3;
    if bound > 0.3 {
        return (bound, bound);
    }
    let mut hair = 1e6f32;
    let mut bead = 1e6f32;
    for index in 0..16 {
        if index >= hairs {
            break;
        }
        let around = turn + index as f32 * TAU / hairs as f32;
        let out = vec3(around.cos() * spread, -1.0, around.sin() * spread).normalize();
        // Each hair droops a little further out along its length under its own weight.
        let elbow = out * reach * 0.55 + vec3(0.0, -reach * 0.08, 0.0);
        let tip = out * reach + vec3(out.x, 0.0, out.z) * reach * 0.15;
        hair = hair.min(
            tapered_tail(p, Vec3::ZERO, elbow, 0.05, 0.03)
                .min(tapered_tail(p, elbow, tip, 0.03, 0.012)),
        );
        bead = bead.min((p - tip).length() - 0.06);
    }
    (hair, bead)
}

/// A strand of kelp.
///
/// A long, fibrous stipe weaving through the currents it spans, wound round by a thin spiral vane
/// that turns the updraft into lift along its whole length as a seed's wing does. At its nodes
/// hang chains of glassy gas beads, and beneath them fans of fine charged hairs reach down towards
/// the aur to gather what drifts up through the forest. A clawed holdfast at its foot knots it to
/// the strands around it into a forest. No two strands grow alike: each has its own length and
/// weave, its vane broad or narrow, tight or open or missing, more beads or fewer, and fans full or
/// sparse, long or short, and some fork into side strands of their own.
pub fn surface(p: Vec3, c: &Plant) -> Surface {
    let length = 9.0 + gene(c, 4.0) * 4.5;
    let thick = 0.14 + gene(c, 5.0) * 0.14;
    let along = p.z.clamp(-length, length);
    let axis = strand(c, along);
    let around = p - axis;
    let mut shape = Surface::new(around.length() * 0.7 - thick, 0);
    // The spiral vane, a thin membrane wound round the stipe, broad or narrow and tight or open as
    // the strand's genes have it; a few strands grow none.
    let vane = gene(c, 10.0);
    if vane > 0.15 {
        let pitch = 1.4 + gene(c, 15.0) * 2.2;
        let width = 0.3 + vane * 0.9;
        let radial = vec2(around.x, around.y).length();
        let angle = around.y.atan2(around.x);
        let turns = p.z / pitch - angle / TAU;
        let gap = (turns - turns.round()).abs() * pitch;
        let edge = (radial - thick - width * 0.5).abs() - width * 0.5;
        let sheet = (gap - 0.03).max(edge).max(p.z.abs() - length + 0.6) * 0.6;
        shape.blend(sheet, 1, 0.08);
    }
    let spacing = 2.2 + gene(c, 6.0) * 1.4;
    let beads = count(c, 7.0, 1, 4);
    let hairs = count(c, 8.0, 7, 15);
    let reach = 0.9 + gene(c, 9.0) * 1.1;
    let spread = 0.5 + gene(c, 3.0) * 0.9;
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
        shape.blend(ellipsoid(p - root, vec3(0.34, 0.34, 0.42) * size), 0, 0.15);
        // A chain of gas beads rising from the node, each smaller than the one below.
        let lift = p - root - vec3(0.0, 0.65, 0.0) * size;
        let mut float = ellipsoid(lift, vec3(0.45, 0.6, 0.45) * size);
        for bead in 1..4 {
            if bead >= beads {
                break;
            }
            let b = bead as f32;
            let shrink = 1.0 - b * 0.2;
            let at = lift - vec3(0.0, b * 0.95 * size, 0.0);
            float = smooth_min(
                float,
                ellipsoid(at, vec3(0.34, 0.4, 0.34) * size * shrink),
                0.12,
            );
        }
        shape.blend(float, 3, 0.1);
        let sway = (c.animation.z * 0.6 + z).sin() * 0.15;
        let (hair, glow) = fan(
            p - root + vec3(0.0, 0.25, 0.0) * size,
            hairs,
            reach * size,
            spread + sway,
            node * 0.7,
        );
        shape.blend(hair, 1, 0.05);
        shape.join(glow, 2);
    }
    // A side strand forking from the main one, its own fan at its tip.
    for branch in 0..count(c, 11.0, 0, 2) {
        let b = branch as f32;
        let from = strand(c, (b - 0.5) * length * 0.6);
        let side = if branch == 0 { 1.0 } else { -1.0 };
        let tip = from + vec3(side * (3.5 + gene(c, 12.0 + b) * 3.0), 1.0, 2.5);
        shape.blend(tapered_tail(p, from, tip, thick * 0.9, thick * 0.5), 0, 0.2);
        shape.blend(
            ellipsoid(p - tip - vec3(0.0, 0.4, 0.0), vec3(0.35, 0.5, 0.35)),
            3,
            0.1,
        );
        let (hair, glow) = fan(p - tip, hairs / 2 + 3, reach * 0.7, spread, b);
        shape.blend(hair, 1, 0.05);
        shape.join(glow, 2);
    }
    // The holdfast's claws, which grip the next strand.
    let foot = strand(c, -length);
    for claw in 0..count(c, 13.0, 4, 7) {
        let angle = claw as f32 * 1.4 + gene(c, 14.0) * 3.0;
        let out = vec3(angle.cos(), angle.sin(), 0.0);
        let bend = foot + out * 0.9 - vec3(0.0, 0.0, 0.7);
        shape.blend(tapered_tail(p, foot, bend, 0.16, 0.1), 0, 0.1);
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
// Glassy, unpigmented membranes, for it feeds on aur: pale horn where they are thick and nearly
// clear where thin, veined with the charge they carry.
pub const LOOK: Look = Look::plant(vec3(0.66, 0.6, 0.5), vec3(0.35, 0.85, 0.9));

pub fn render(ray: Ray, c: &Plant) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
