use crate::scene::life::{
    CreatureInstance, Look,
    animalia::bullidorsata::caelomorpha::{GasChamber, drift_body, tissue_distance},
    biology::Biology,
    rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 3650.0,
    lifespan_days: 146_000.0,
    adult_mass: 120.0,
    altitude: (50.0, 2500.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 46.0;

taxon!(
    Species,
    "Aerogigas auratus",
    "Air Giants",
    [
        "Air Giants drift through the humid lower Gardens beneath broad golden canopies, their vaulted undersides enclosing sheltered pockets of air.",
        "Large gas chambers support the crown while smaller chambers keep the scalloped fringe spread through the updraft.",
        "Amber veins and pale branching ribs frame translucent tissues stippled with intercepted algae and plant fragments.",
        "Feeding folds face into the passing current, so a slow change in tilt brings fresh food through the arches without powered flight.",
        "During the dim phase of the aur cycle, the canopy settles lower and its moist underside becomes a refuge for smaller drifters.",
    ],
);

/// The canopy's lift: a ring of chambers in the crown's thickness around a
/// larger one at its heart, so a torn chamber costs only part of the lift.
const CHAMBERS: [GasChamber; 6] = [
    GasChamber {
        center: vec3(0.0, 3.7, 0.0),
        radii: vec3(3.6, 1.1, 3.6),
    },
    GasChamber {
        center: vec3(6.6, 2.9, 0.0),
        radii: vec3(2.5, 0.95, 2.5),
    },
    GasChamber {
        center: vec3(2.04, 2.9, 6.28),
        radii: vec3(2.5, 0.95, 2.5),
    },
    GasChamber {
        center: vec3(-5.34, 2.9, 3.88),
        radii: vec3(2.5, 0.95, 2.5),
    },
    GasChamber {
        center: vec3(-5.34, 2.9, -3.88),
        radii: vec3(2.5, 0.95, 2.5),
    },
    GasChamber {
        center: vec3(2.04, 2.9, -6.28),
        radii: vec3(2.5, 0.95, 2.5),
    },
];

/// Ribs and the hollows between them, around the canopy.
const SECTOR: f32 = TAU / 10.0;

/// The canopy's outline at an angle: a scalloped fringe.
fn rim(angle: f32) -> f32 {
    15.5 * (1.0 + (angle * 10.0).cos() * 0.045)
}

/// The canopy's upper surface and its thickness at a share of the way out
/// to the fringe, rising and falling slowly as the chambers breathe.
fn shell(reach: f32, c: &CreatureInstance) -> (f32, f32) {
    let breathe = (c.animation.z * 0.4).sin() * 0.15;
    let crown = 1.0 - reach * reach;
    ((4.6 + breathe) * crown, 0.35 + 2.4 * crown)
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(17.0, 9.0, 17.0).extend(0.0);
}

/// The organs: a gelatinous mass at the heart of the crown, and amber ribs
/// branching out through the canopy's thickness to its fringe, holding the
/// vaults open beneath.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let angle = p.z.atan2(p.x);
    let core = rounded(p - vec3(0.0, 3.2, 0.0), vec3(3.0, 0.8, 3.0));
    let turn = (angle / SECTOR).round() * SECTOR;
    let (sine, cosine) = turn.sin_cos();
    let along = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let reach = (along / rim(turn)).clamp(0.0, 1.0);
    let (top, thickness) = shell(reach, c);
    let rib = (vec2(across, p.y - (top - thickness * 0.5)).length() - 0.3 * (1.0 - reach * 0.7))
        .max(along - rim(turn) * 0.95)
        .max(-along);
    Surface::new(smooth_min(core, rib, 0.8) + tissue_distance(p, 12.0), 0)
}

/// The canopy: a broad, low dome of gel, thickest at the crown and
/// thinning to a scalloped fringe; its underside vaulted up into arches
/// between the ribs, each hollow sheltering a curtain of feeding folds from
/// drying out and from sudden gusts, rippling in the air drawn through it.
pub fn envelope(p: Vec3, c: &CreatureInstance) -> Surface {
    let radial = vec2(p.x, p.z).length();
    let angle = p.z.atan2(p.x);
    let edge = rim(angle);
    let reach = (radial / edge).min(1.2);
    let (top, thickness) = shell(reach, c);
    let ribs = (angle * 5.0).cos().abs();
    let vault = (1.0 - ribs).sqrt()
        * 1.6
        * reach.smoothstep(0.15, 0.55)
        * (1.0 - reach.smoothstep(0.85, 1.0));
    // Never vaulting through the canopy itself, however thin it grows.
    let bottom = (top - thickness + vault).min(top - 0.3);
    let dome = (p.y - top).max(bottom - p.y).max(radial - edge) * 0.5;
    let hollow = ((angle - SECTOR * 0.5) / SECTOR).round() * SECTOR + SECTOR * 0.5;
    let (sine, cosine) = hollow.sin_cos();
    let along = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let out = (along / edge).clamp(0.0, 1.0);
    let (fold_top, fold_thickness) = shell(out, c);
    let roof = (fold_top - fold_thickness + 1.2 * out.smoothstep(0.15, 0.55)).min(fold_top - 0.3);
    let drop = 3.0 * (out * PI).sin();
    let ripple = (along * 1.1 - c.animation.z * 0.8).sin() * 0.3 * (roof - p.y).max(0.0) / 3.0;
    let fold = ((across - ripple).abs() - 0.14)
        .max(p.y - roof)
        .max(roof - drop - p.y)
        .max((along - edge * 0.52).abs() - edge * 0.34)
        * 0.6;
    Surface::new(smooth_min(dome, fold, 0.5) + tissue_distance(p, 12.0), 0)
}

/// Tint and scale, shared by the preview and the world renderer.
pub const LOOK: Look = Look::drift(vec3(0.78, 0.65, 0.30), 12.0);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    drift_body(ray, c, envelope, surface, LOOK.accent, 12.0, &CHAMBERS)
}
