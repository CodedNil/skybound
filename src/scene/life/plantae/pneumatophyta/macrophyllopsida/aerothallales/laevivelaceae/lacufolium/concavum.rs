use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, gene, pick, speckle},
    smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::{
        physiology::{BLADDER, FROND, ROOT, STEM},
        pneumatophyta::macrophyllopsida::aerothallales::laevivelaceae::lacufolium::BASIN,
    },
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.1,
    maturity_days: 1200.0,
    lifespan_days: 90000.0,
    adult_mass: 30.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 15.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Lacufolium concavum",
    "Pondplants",
    [
        "Pondplants curve a broad floating leaf into a shallow basin supported by a flexible outer rim.",
        "Rain accumulates above the collecting membrane, providing drinking water for animals whose moisture-recovery organs cannot meet the cost of active flight.",
        "The basin flexes as its load changes, spilling excess water before the plant loses safe lift.",
        "Hygroscopic tissues supply the plant itself, while the pond remains an exposed resource visited by drinkers, cleaners and opportunistic parasites.",
        "Grazing damage drains this shared water store, so frequent visitors benefit from keeping the living rim intact.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(9.0, 5.0, 9.0).extend(0.0);
}

/// The bottom of the basin, where the plant's hub hangs beneath it.
const FLOOR: f32 = -1.8;

/// The basin's edge at a bearing `angle`, as a share of its full reach. Basins grow round,
/// scalloped where the rim's ribs end, notched where an old spill wore a gap, or drawn out to a
/// drip tip that sheds the overflow over one side.
fn outline(c: &CreatureInstance, angle: f32) -> f32 {
    let turn = angle + gene(c, 2.0) * TAU;
    match pick(c, 1.0, 4) {
        0 => 1.0,
        1 => 0.9 + 0.1 * (turn * count(c, 3.0, 5, 9) as f32).cos().abs(),
        2 => 1.0 - 0.3 * (1.0 - (turn.cos() * 0.5 + 0.5)).powi(6),
        _ => 0.82 + 0.18 * turn.cos().powi(2) + 0.25 * turn.cos().max(0.0).powi(8),
    }
}

/// A pondplant.
///
/// A broad leaf cupped into a basin that holds the rain, ringed by a rim that keeps it in, with a
/// knotted hub beneath it from which its floats and roots hang. Each grows its own: deep or
/// shallow, full or nearly dry, round, scalloped, notched or spouted, with a rolled, frilled or
/// thorned rim, a ring of floats under the rim or one great float under the hub, and few or many
/// roots trailing below.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let reach = 5.0 + gene(c, 4.0) * 2.8;
    let depth = 0.8 + gene(c, 5.0) * 1.4;
    let along = p.xz().length();
    let angle = p.z.atan2(p.x);
    let edge = reach * outline(c, angle);
    let rise = |r: f32| FLOOR + depth * (r / edge).powi(2);
    let slope = 2.0 * depth * along / (edge * edge);
    let shell = (p.y - rise(along)).abs() / (1.0 + slope * slope).sqrt() - 0.12;
    let mut shape = Surface::new(shell.max(along - edge), 1);
    // The water it holds, as full as the last rain left it.
    let level = FLOOR + depth * (0.4 + gene(c, 6.0) * 0.55);
    let water = (p.y - level).max(rise(along) - p.y).max(along - edge);
    shape.join(water, 4);
    // The rim.
    let lip = vec2(along - edge, p.y - rise(edge));
    let rim = match pick(c, 7.0, 3) {
        0 => lip.length() - 0.28,
        1 => {
            let frill = (angle * (12.0 + gene(c, 8.0) * 10.0)).sin() * 0.12;
            let wall = 0.4 + gene(c, 9.0) * 0.5;
            // A thin wall standing up from the edge, ruffled along its length.
            ((lip.x - frill).abs() - 0.06).max((lip.y - wall * 0.5).abs() - wall * 0.5)
        }
        _ => {
            let spines = 14.0 + gene(c, 10.0) * 14.0;
            let sector = TAU / spines;
            let at = (angle / sector).round() * sector;
            let base = vec3(at.cos() * edge, rise(edge), at.sin() * edge);
            let out = vec3(at.cos(), 0.35, at.sin());
            (lip.length() - 0.18).min(tapered_tail(p, base, base + out * 0.9, 0.08, 0.01))
        }
    };
    // The rim grows out of the leaf, not onto it.
    shape.blend(rim, 3, 0.3);
    // The hub, and the floats that hold the basin up.
    shape.join(
        ellipsoid(p - vec3(0.0, FLOOR - 0.5, 0.0), vec3(1.1, 0.55, 1.1)),
        0,
    );
    if pick(c, 11.0, 2) == 0 {
        let floats = count(c, 12.0, 5, 9);
        let sector = TAU / floats as f32;
        let at = (angle / sector).round() * sector;
        let out = edge * 0.72;
        let centre = vec3(at.cos() * out, rise(out) - 0.6, at.sin() * out);
        shape.blend(
            capsule(p, centre, vec3(0.0, FLOOR - 0.5, 0.0), 0.12),
            0,
            0.15,
        );
        shape.blend(ellipsoid(p - centre, vec3(0.7, 0.5, 0.7)), 3, 0.25);
    } else {
        shape.join(
            ellipsoid(p - vec3(0.0, FLOOR - 1.4, 0.0), vec3(1.6, 1.1, 1.6)),
            3,
        );
    }
    // Roots trailing beneath, drawing water and aur from the air below.
    let roots = count(c, 13.0, 3, 9);
    let long = 2.0 + gene(c, 14.0) * 3.5;
    for i in 0..roots {
        let f = i as f32;
        let bearing = f * TAU / roots as f32 + speckle(vec2(f, 1.0), 1.0);
        let out = vec3(bearing.cos(), 0.0, bearing.sin());
        let root = vec3(0.0, FLOOR - 0.6, 0.0) + out * 0.6;
        let sway = (c.animation.z * 0.5 + f).sin() * 0.3;
        let knee = root + out * 0.8 + vec3(sway, -long * 0.5, 0.0);
        let tip = knee + out * 0.4 + vec3(sway, -long * 0.5, 0.0);
        shape.blend(
            smooth_min(
                tapered_tail(p, root, knee, 0.14, 0.09),
                tapered_tail(p, knee, tip, 0.09, 0.02),
                0.1,
            ),
            0,
            0.15,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -2.30, 0.00), vec3(0.40, 0.20, 0.40)),
    site(BLADDER, vec3(0.00, -2.35, 0.50), vec3(0.20, 0.15, 0.20)),
    pair(FROND, vec3(1.00, -1.78, 0.00), vec3(0.08, 0.05, 0.08)),
    site(ROOT, vec3(0.00, -2.65, 0.00), vec3(0.20, 0.10, 0.20)),
    site(BASIN, vec3(0.00, -1.55, 0.00), vec3(0.20, 0.05, 0.20)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.24, 0.5, 0.17), vec3(0.65, 0.79, 0.3));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
