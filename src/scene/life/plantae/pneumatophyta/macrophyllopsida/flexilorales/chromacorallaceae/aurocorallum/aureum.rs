use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{count, gene, pick, speckle},
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.43,
    maturity_days: 900.0,
    lifespan_days: 30000.0,
    adult_mass: 1.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 13.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Aurocorallum aureum",
    "Golden Coral",
    [
        "Golden Coral grows layered golden fans from branching attachments on large, slow-moving animals.",
        "Frilled surfaces collect dispersed aur while the host carries the colony through fresh air.",
        "The fans yield in a changing current, reducing strain on their attachments and avoiding the weight of a separate buoyant body.",
        "Moist recesses between branches retain nutrient-bearing dust and shelter small scavengers.",
        "The association remains most stable where modest growth leaves the host's breathing folds and lifting surfaces clear.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

/// Where the colony holds on to its host.
const FOOT: Vec3 = vec3(0.0, -1.9, 0.0);

/// A fan of one of the forms the colony grows, `p` relative to where it springs from its stalk,
/// opening `out` from the stalk: a frilled shelf, a funnel, an upright paddle, or a ruffled ruff
/// spiralling round the stalk.
fn fan(p: Vec3, out: Vec3, form: u32, size: f32, frill: f32) -> f32 {
    let centre = out * size * 0.8;
    let q = p - centre;
    let along = q.xz().length();
    let angle = q.z.atan2(q.x);
    let ruffle = (angle * frill).sin() * 0.12 * along / size;
    match form {
        0 => ((q.y - ruffle - along * 0.15).abs() - 0.05).max(along - size),
        1 => ((q.y - along * 0.7 - ruffle).abs() * 0.8 - 0.05).max(along - size),
        2 => {
            let across = vec3(out.z, 0.0, -out.x);
            let flat = q.dot(across);
            let rise = vec2(q.dot(out), q.y - size * 0.6);
            ((flat - ruffle).abs() - 0.05).max((rise / vec2(1.0, 1.6)).length() - size)
        }
        _ => {
            let turn = (angle + q.y * 2.0).sin() * 0.25;
            ((q.y - turn - ruffle).abs() - 0.05)
                .max(along - size)
                .max(0.3 - along)
        }
    }
}

/// A colony of golden coral.
///
/// It holds on to a slow animal's hide by a broad foot, from which branching stalks rise and
/// lean out, each carrying fans that open in tiers into the air the host carries them through.
/// Colonies grow their own way: a few stalks or many, fans as shelves, funnels, paddles or ruffs,
/// one tier to a stalk or a stack of them, frilled finely or coarsely.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - FOOT, vec3(1.3, 0.5, 1.3)), 0);
    let stalks = count(c, 1.0, 2, 5);
    let form = pick(c, 2.0, 4);
    let tiers = count(c, 3.0, 1, 4);
    let frill = 6.0 + gene(c, 4.0) * 14.0;
    let size = (1.9 - tiers as f32 * 0.25) * (0.8 + gene(c, 5.0) * 0.4);
    for i in 0..stalks {
        let f = i as f32;
        let bearing = f * TAU / stalks as f32 + gene(c, 6.0) * TAU;
        let out = vec3(bearing.cos(), 0.0, bearing.sin());
        let lean = 0.3 + speckle(vec2(f, 1.0), 1.0) * 0.6;
        let tall = 3.0 + speckle(vec2(f, 2.0), 1.0) * 2.5;
        let top = FOOT + out * tall * lean + vec3(0.0, tall, 0.0);
        let bend = FOOT + out * tall * lean * 0.3 + vec3(0.0, tall * 0.55, 0.0);
        shape.join(
            tapered_tail(p, FOOT, bend, 0.32, 0.22).min(tapered_tail(p, bend, top, 0.22, 0.1)),
            0,
        );
        for tier in 0..tiers {
            let t = (tier + 1) as f32 / tiers as f32;
            let at = if t < 0.55 {
                FOOT.lerp(bend, t / 0.55)
            } else {
                bend.lerp(top, (t - 0.55) / 0.45)
            };
            let turn = bearing + tier as f32 * 1.3;
            let facing = vec3(turn.cos(), 0.0, turn.sin());
            shape.join(fan(p - at, facing, form, size * (1.1 - t * 0.3), frill), 3);
        }
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.90, 0.00), vec3(0.40, 0.18, 0.40)),
    site(BLADDER, vec3(0.00, -1.80, -0.60), vec3(0.18, 0.12, 0.18)),
    pair(FROND, vec3(0.70, -1.90, 0.10), vec3(0.18, 0.12, 0.18)),
    site(ROOT, vec3(0.00, -2.15, 0.40), vec3(0.18, 0.10, 0.18)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.75, 0.42, 0.07), vec3(1.0, 0.7, 0.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
