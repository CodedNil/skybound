use crate::scene::life::{
    Look, Plant,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, gene, pick, speckle},
    smooth_min,
    surface::Surface,
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Part, pair, site},
    plantae::{
        physiology::{BLADDER, FROND, ROOT, STEM},
        pneumatophyta::macrophyllopsida::aerothallales::laevivelaceae::lacufolium::BASIN,
    },
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.1,
    maturity_days: 1200.0,
    lifespan_days: 90000.0,
    adult_mass: 30.0,
    altitude: (1500.0, 6000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 15.0;

taxon!(
    Species,
    "Lacufolium concavum",
    "Pondplants",
    [
        "Pondplants stretch a membrane over a ring of upturned ribs, and it sags between them into a pleated, star-shaped basin; many raise a combed mast from the hub that wrings water from passing cloud.",
        "Rain accumulates above the collecting membrane, providing drinking water for animals whose moisture-recovery organs cannot meet the cost of active flight.",
        "The basin flexes as its load changes, spilling excess water before the plant loses safe lift.",
        "Hygroscopic tissues supply the plant itself, while the pond remains an exposed resource visited by drinkers, cleaners and opportunistic parasites.",
        "Grazing damage drains this shared water store, so frequent visitors benefit from keeping the living ribs and membrane intact.",
    ],
);

pub fn animate(c: &mut Plant) {
    c.opaque_bounds = vec3(9.0, 5.0, 9.0).extend(0.0);
}

/// The bottom of the basin, where the plant's hub hangs beneath it.
const FLOOR: f32 = -1.8;

/// A pondplant.
///
/// Not a leaf but a frame: ribs radiate from a knotted hub like the spokes of an umbrella turned
/// up to the sky, and a membrane stretched between them sags into deep pleats under the water's
/// weight, so the pool it holds is a star and the rim between the ribs dips in scallops where the
/// overflow spills. The ribs run on past the membrane and curl up at their tips. Many grow a mast
/// from the hub, ringed with fine combs that wring water from the cloud blowing through them and
/// drip it into the pool. Each grows its own: few ribs or many, deep or shallow, taut or slack, a
/// mast or none, a ring of floats under the rim or one great float under the hub, and few or many
/// roots trailing below.
pub fn surface(p: Vec3, c: &Plant) -> Surface {
    let reach = 5.0 + gene(c, 4.0) * 2.8;
    let depth = 0.8 + gene(c, 5.0) * 1.4;
    let ribs = count(c, 1.0, 6, 11) as f32;
    let sag = 0.25 + gene(c, 2.0) * 0.55;
    let along = p.xz().length();
    let angle = p.z.atan2(p.x);
    let sector = TAU / ribs;
    let nearest = (angle / sector).round() * sector;
    // How far between two ribs the bearing lies, from on a rib to midway between.
    let between = ((angle - nearest) / sector).abs() * 2.0;
    let pleat = (between * PI * 0.5).sin();
    let edge = reach * (1.0 - 0.14 * pleat);
    let rib = |r: f32| FLOOR + depth * (r / reach).max(0.0).powf(1.6);
    let rise =
        |r: f32| rib(r) - sag * pleat * (r / reach).min(1.0) * (1.0 - (r / reach).min(1.0) * 0.4);
    let slope = 1.6 * depth * (along / reach).max(1e-3).powf(0.6) / reach;
    let shell = (p.y - rise(along)).abs() / (1.0 + slope * slope).sqrt() - 0.08;
    let mut shape = Surface::new(shell.max(along - edge), 1);
    // The water it holds, as full as the rain and its drinkers have left it: not inherited but its
    // life's, so a basin by a busy watering place runs low and one in wet cloud brims. The membrane
    // sags into a sump that always keeps a little, which drinkers cannot reach.
    let level = FLOOR + depth * (0.35 + c.water.x.clamp(0.0, 1.0) * 0.6);
    let water = (p.y - level).max(rise(along) - p.y).max(along - edge);
    shape.join(water, 4);
    // The nearest rib, from the hub out past the membrane's edge, curling up at its tip.
    let spoke = vec3(nearest.cos(), 0.0, nearest.sin());
    let at = |r: f32| spoke * r + Vec3::Y * rib(r);
    let tip = at(reach * 1.12) + Vec3::Y * (0.5 + gene(c, 8.0) * 0.9);
    let spine = tapered_tail(p, at(0.4), at(reach * 0.6), 0.16, 0.12)
        .min(tapered_tail(
            p,
            at(reach * 0.6),
            at(reach * 1.02),
            0.12,
            0.08,
        ))
        .min(tapered_tail(p, at(reach * 1.02), tip, 0.08, 0.02));
    shape.blend(spine, 0, 0.12);
    // The mast and its combs.
    if gene(c, 9.0) > 0.3 {
        let height = 1.5 + gene(c, 10.0) * 2.5;
        let top = vec3(0.0, FLOOR + depth + height, 0.0);
        let mast = tapered_tail(p, vec3(0.0, FLOOR, 0.0), top, 0.18, 0.05);
        let spacing = 0.35 + gene(c, 3.0) * 0.25;
        let ring = ((p.y - FLOOR - depth * 0.6) / spacing).round();
        let ring_y = FLOOR + depth * 0.6 + ring * spacing;
        let span =
            (1.0 - (ring_y - FLOOR) / (top.y - FLOOR)).clamp(0.0, 1.0) * (0.5 + gene(c, 7.0) * 0.6);
        let comb = (vec2(along - span * 0.5, p.y - ring_y).length() - 0.025)
            .max(along - span)
            .max(ring_y - top.y + 0.2)
            .max(FLOOR + depth * 0.5 - ring_y);
        shape.blend(mast.min(comb), 0, 0.06);
    }
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

/// Where its organs lie in its body, each placed by its own design; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[
    site(STEM, vec3(0.00, -2.30, 0.00), vec3(0.40, 0.20, 0.40)),
    site(BLADDER, vec3(0.00, -2.35, 0.50), vec3(0.20, 0.15, 0.20)),
    pair(FROND, vec3(1.00, -1.78, 0.00), vec3(0.08, 0.05, 0.08)),
    site(ROOT, vec3(0.00, -2.65, 0.00), vec3(0.20, 0.10, 0.20)),
    site(BASIN, vec3(0.00, -1.55, 0.00), vec3(0.20, 0.05, 0.20)),
];

/// Colouring and wings, shared by the preview and the world renderer.
// Bone-pale and unpigmented, for it feeds on aur.
pub const LOOK: Look = Look::plant(vec3(0.62, 0.6, 0.55), vec3(0.4, 0.8, 0.9));
