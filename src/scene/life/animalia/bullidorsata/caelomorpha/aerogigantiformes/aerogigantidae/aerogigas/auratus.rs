use crate::scene::life::{
    CreatureInstance, Look,
    animalia::bullidorsata::caelomorpha::{GasChamber, drift_body, tissue_distance},
    biology::{Biology, Litter},
    genome::Gene,
    rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    organs::{Viscera, pair, site},
};
use core::f32::consts::{PI, TAU};
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.75,
    maturity_days: 3650.0,
    lifespan_days: 146_000.0,
    adult_mass: 120.0,
    altitude: (50.0, 2500.0),
    // Spawn released in clouds of a few to a dozen.
    litter: Litter::new(2, 2, [0.2, 0.3, 0.25, 0.15, 0.07, 0.03]),
    gestation_days: 60.0,
};

pub const RENDER_RADIUS: f32 = 46.0;

/// How high its crown is domed, how deeply its fringe is scalloped, and how far its feeding folds
/// hang.
///
/// A high dome holds more gas, a deep scallop spills gusts before they can flip it, and long folds
/// sieve more algae from the air drawn through them.
pub const DOME: Gene = Gene(38);
pub const SCALLOP: Gene = Gene(39);
pub const FOLDS: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Aerogigas auratus",
    "Air Giants",
    [
        "Air Giants drift through the humid lower Gardens beneath broad golden canopies, their vaulted undersides enclosing sheltered pockets of air.",
        "Large gas chambers support the crown while smaller chambers keep the scalloped fringe spread through the updraft.",
        "Amber veins and pale branching ribs frame translucent tissues stippled with intercepted algae and plant fragments.",
        "Feeding folds face into the passing current, so a slow change in tilt brings fresh food through the arches without powered flight.",
        "Where the updraft slackens, the canopy settles lower and its moist underside becomes a refuge for smaller drifters.",
    ],
);

/// The canopy's lift: a ring of chambers in the crown's thickness around a larger one at its heart,
/// so a torn chamber costs only part of the lift.
#[expect(
    clippy::approx_constant,
    reason = "6.28 is where two chambers sit on the ring, not a turn"
)]
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
fn rim(angle: f32, c: &CreatureInstance) -> f32 {
    15.5 * (1.0 + (angle * 10.0).cos() * (0.02 + c.gene(SCALLOP) * 0.06))
}

/// The canopy's upper surface and its thickness at a share of the way out to the fringe, rising and
/// falling slowly as the chambers breathe.
fn shell(reach: f32, c: &CreatureInstance) -> (f32, f32) {
    let breathe = (c.animation.z * 0.4).sin() * 0.15;
    let crown = 1.0 - reach * reach;
    (
        (3.8 + c.gene(DOME) * 1.6 + breathe) * crown,
        0.35 + 2.4 * crown,
    )
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(17.0, 9.0, 17.0).extend(0.0);
}

/// The organs: a gelatinous mass at the heart of the crown, and amber ribs branching out through
/// the canopy's thickness to its fringe, holding the vaults open beneath.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let angle = p.z.atan2(p.x);
    let core = rounded(p - vec3(0.0, 3.2, 0.0), vec3(3.0, 0.8, 3.0));
    let turn = (angle / SECTOR).round() * SECTOR;
    let (sine, cosine) = turn.sin_cos();
    let along = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let reach = (along / rim(turn, c)).clamp(0.0, 1.0);
    let (top, thickness) = shell(reach, c);
    let rib = (vec2(across, p.y - (top - thickness * 0.5)).length() - 0.3 * (1.0 - reach * 0.7))
        .max(along - rim(turn, c) * 0.95)
        .max(-along);
    Surface::new(smooth_min(core, rib, 0.8) + tissue_distance(p, 12.0), 0)
}

/// The canopy: a broad, low dome of gel.
///
/// It is thickest at the crown and thins to a scalloped fringe; its underside is vaulted up into
/// arches between the ribs, each hollow sheltering a curtain of feeding folds from drying out and
/// from sudden gusts, rippling in the air drawn through it.
pub fn envelope(p: Vec3, c: &CreatureInstance) -> Surface {
    let radial = vec2(p.x, p.z).length();
    let angle = p.z.atan2(p.x);
    let edge = rim(angle, c);
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
    let drop = (2.3 + c.gene(FOLDS) * 1.4) * (out * PI).sin();
    let ripple = (along * 1.1 - c.animation.z * 0.8).sin() * 0.3 * (roof - p.y).max(0.0) / 3.0;
    let fold = ((across - ripple).abs() - 0.14)
        .max(p.y - roof)
        .max(roof - drop - p.y)
        .max((along - edge * 0.52).abs() - edge * 0.34)
        * 0.6;
    Surface::new(smooth_min(dome, fold, 0.5) + tissue_distance(p, 12.0), 0)
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(FLOAT, vec3(0.00, 3.13, -0.46), vec3(0.85, 0.60, 0.85)),
    pair(FLOAT, vec3(2.73, 3.05, -0.30), vec3(0.21, 0.21, 0.21)),
    site(CROP, vec3(0.00, 3.13, 1.06), vec3(0.95, 0.43, 0.95)),
    pair(BREATHER, vec3(2.73, 3.05, 0.30), vec3(0.17, 0.17, 0.17)),
    site(BROOD, vec3(0.00, 3.13, 2.28), vec3(0.32, 0.20, 0.32)),
]);

/// Tint and scale, shared by the preview and the world renderer.
pub const LOOK: Look = Look::drift(vec3(0.78, 0.65, 0.30), 12.0);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    drift_body(ray, c, envelope, surface, LOOK.accent, 12.0, &CHAMBERS)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! An Air Giant cannot fly, but it can steer by tilting: a slow lean brings its feeding folds
    //! square to the passing current, so the air pours its algae through the arches, and when the
    //! air is still it drifts towards the strongest rising air it can find.

    use crate::scene::life::{
        animalia::{
            bullidorsata::ETHOLOGY as GAS_BAG,
            mind::{Ethology, Intent, Situation, Urges},
            nature::RIDE,
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology { urges, ..GAS_BAG };

    /// Leaning the canopy so its feeding folds face into the current.
    const TILT: Intent = Intent {
        name: "tilt",
        act: |_, s| {
            Goal::new(
                Activity::Feed,
                Some(s.seen.wind.normalize_or_zero() * 30.0),
                0.15,
            )
            .hovering()
        },
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (GAS_BAG.urges)(s, urges);
        if s.seen.updraft < 0.5 {
            urges.add(RIDE, 0.3);
        }
        let current = s.seen.wind.length();
        if current > 2.0 {
            urges.add(TILT, s.drives.hunger * 0.8 * (current / 10.0).min(1.0));
        }
    }
}
