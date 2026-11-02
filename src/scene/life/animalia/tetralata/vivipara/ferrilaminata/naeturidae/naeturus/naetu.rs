use super::{shaft, vane};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::tetralata::vivipara::ferrilaminata::naeturidae::AUR_SAC,
    organs::{Part, pair},
};
use crate::scene::{
    life::{
        Look, Tetralate,
        animalia::{
            tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
            visit_wings, with_wings,
        },
        biology::{Biology, Litter},
        ellipsoid,
        genome::Gene,
        louvres, prepare_pose, rounded, smooth_min,
        surface::{
            FURRED, Hit, Material, Ray, Surface, TraceResult, WRINKLED, local_ray, make_hit,
            trace_material,
        },
        tapered_tail,
    },
    math::{Power, intersect_sphere},
};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.7,
    maturity_days: 7300.0,
    lifespan_days: 180_000.0,
    adult_mass: 150.0,
    // A single calf after a long carrying; twins are rare.
    litter: Litter::new(1, 1, [0.94, 0.06, 0.0, 0.0, 0.0, 0.0]),
    gestation_days: 540.0,
    altitude: (300.0, 12000.0),
};

pub const RENDER_RADIUS: f32 = 70.0;

/// How full its gas lobes bulge, how large its aur sacs, how far its exchange frills fan, and how
/// deep the fibres of its back grow.
///
/// Full lobes carry more and lose less to a wound, large sacs hold more of the mists' charge and
/// raise a stronger shield, long frills breathe more of the air it flies through, and a deep pelt
/// holds the Næ's dwellings fast.
pub const CHAMBERS: Gene = Gene(38);
pub const AUR_SACS: Gene = Gene(39);
pub const FRILLS: Gene = Gene(40);
pub const FIBRES: Gene = Gene(41);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Species,
    "Naetu",
    "Nætu",
    [
        "Nætu are enormous four-winged carriers with broad flat backs, a blunt head under a heavy brow, external exchange frills and a long tail that spreads into a fan of feather vanes.",
        "Three pairs of mouth tentacles gather vegetation, while redundant gas lobes along the upper flanks and pleated breathing channels reduce the cost of slow flight; the belly is pleated lengthwise round the great crop, and the folds open as it fills.",
        "Their wings are broad and stalked thick, for a carrier cruises rather than hovers, and beat slowly enough to count.",
        "A raised ridge rings the back, and in it sit the aur sacs, sheathed in iron: from its twentieth year a Nætu can let their charge out along the ridge as a shimmering dome over its back, brightest where it leaves the ridge, which turns the wind and the spray of the mists from everything that lives there.",
        "Within the ring grow the grassy dorsal fibres that anchor Næ dwellings.",
        "Carefully controlled dives skim the mists above the aur ocean to recharge the sacs, sustaining healing and the dome without contact with pure aur.",
        "Recognition through touch, grooming and vocal exchanges binds travelling herds to the Næ families that care for them.",
        "They eat metal as all their order does, cropping iron-rich moss and scraping mineral crust from the skies they cross, and sheathe each aur sac in it: a conducting lining that holds the charge in and keeps it from the flesh around, thickening through a long life.",
    ],
);

/// The head: its centre and half extents.
const HEAD: Vec3 = vec3(0.0, 0.4, -13.5);
const HEAD_SIZE: Vec3 = vec3(5.6, 4.4, 5.6);

/// Where the mouth opens beneath the head, and its tentacles with it.
const MOUTH: Vec3 = vec3(0.0, -2.1, -17.7);

/// How far the mouth tentacles reach, as a share of the lineage's usual reach.
const REACH: f32 = 2.6;

/// The ridge round the back: its centre along the body, its half spans across and along, and how
/// high it runs, sinking a little towards its ends as the back falls away.
const RING: Vec3 = vec3(5.6, 1.5, 11.0);

fn ring_height(z: f32) -> f32 {
    let along = (z - RING.y) / RING.z;
    5.05 - 0.6 * along * along
}

/// How far round the ring a point lies from its line, in the back's plane: zero on it, negative
/// within.
fn ring_distance(point: Vec3) -> f32 {
    (vec2(point.x / RING.x, (point.z - RING.y) / RING.z).length() - 1.0) * RING.x
}

/// How far into its twentieth year the Nætu has come into its shield, from nothing to whole.
fn shielded(creature: &Tetralate) -> f32 {
    ((creature.life.x - BIOLOGY.maturity_days) / 365.0).clamp(0.0, 1.0)
}

/// A point on the head's surface, `inset` of the way out along `direction` from its centre.
fn on_head(direction: Vec3, inset: f32) -> Vec3 {
    HEAD + HEAD_SIZE * (direction / direction.length()) * inset
}

pub fn animate(creature: &mut Tetralate) {
    prepare_pose(creature, 20.0, 25.0, 0);
    creature.opaque_bounds = vec3(16.0, 14.0, 58.0).extend(0.0);
    rig_tentacles(creature, MOUTH, REACH, ORAL_PAIRS);
}

/// Where its tail runs: straight out of the body, sinking a little and swaying slowly.
fn tail_joint(creature: &Tetralate, index: u32) -> Vec3 {
    let t = index as f32 / 7.0;
    let sway = (creature.animation.z * 1.2 - t * 3.0).sin() * t * 2.0;
    vec3(sway, 0.5 - t * 1.6, 15.0 + t * 30.0)
}

/// A flying island: a broad, flat-backed carrier.
///
/// A blunt head under a heavy brow, its eyes set low to look forwards and down, the exchange
/// frills fanning behind them; a wide body whose back is flat where the dorsal fibres grow and
/// the Næ build, ringed by the raised ridge that holds the aur sacs; gas lobes swelling along the
/// upper flanks so no single wound brings it down; a belly pleated lengthwise round the crop; and a
/// long tail flowing straight out of the body into a fan of feather vanes.
pub fn surface(point: Vec3, creature: &Tetralate) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let head = rounded(point - HEAD, HEAD_SIZE);
    let torso = rounded(point - vec3(0.0, 0.0, -1.0), vec3(7.6, 5.0, 13.0));
    let hind = rounded(point - vec3(0.0, 0.4, 10.0), vec3(5.6, 4.0, 8.0));
    let slab = rounded(point - vec3(0.0, 3.0, 0.0), vec3(6.6, 2.2, 12.5));
    let mut d = smooth_min(smooth_min(torso, hind, 3.0), slab, 2.5);
    d = smooth_min(d, head, 3.0);
    // The back flattened, a smooth shoulder down to the flanks.
    d = -smooth_min(-d, -(point.y - 5.15), 0.8);
    // A heavy brow over the eyes.
    let brow = rounded(point - vec3(0.0, 3.4, -15.0), vec3(5.2, 1.5, 4.4));
    d = smooth_min(d, brow, 1.5);
    // Gas lobes along the upper flanks, each its own size, scalloping the outline from above.
    let lobes = 0.85 + creature.gene(CHAMBERS) * 0.3;
    for index in 0..3 {
        let z = index as f32 * 8.0 - 7.0;
        let size = 1.0 + (index as f32 - 1.0).abs() * -0.1 + if index == 1 { 0.1 } else { 0.0 };
        let lobe = rounded(
            mirrored - vec3(6.2, 2.6, z),
            vec3(2.6, 1.9, 4.0) * lobes * size,
        );
        d = smooth_min(d, lobe, 1.4);
    }
    let tail = shaft(point, |index| tail_joint(creature, index), 8, 3.4, 0.7);
    d = smooth_min(d, tail, 3.0);
    // The belly pleated lengthwise round the crop.
    if point.y < -1.0 && point.x.abs() < 6.5 && point.z > -14.0 && point.z < 12.0 {
        let groove = (1.0 - (point.x * 1.4).sin().abs()).power(6);
        d += groove * 0.18 * ((-1.0 - point.y) / 1.5).clamp(0.0, 1.0);
    }
    // The mouth, a wide slit beneath the head.
    let slit = rounded(point - MOUTH - vec3(0.0, 0.1, 0.4), vec3(2.6, 0.3, 0.9));
    d = d.max(-slit);
    d = louvres(d, point - vec3(0.0, 0.0, -4.0), vec3(8.0, 5.0, 10.0));
    // The ridge round the back, grown up out of it, that the shield is let out along.
    let ring = ring_distance(point);
    let ridge = vec2(ring, point.y - ring_height(point.z)).length() - 0.45;
    d = smooth_min(d, ridge, 0.7);
    // Within the ring, the dorsal fibres.
    let inside = ring < -0.6 && point.y > 4.6;
    if inside {
        let fur = (point.x * 23.0 + (point.z * 7.0).sin() * 2.0).sin()
            * (point.z * 29.0 + (point.x * 3.0).sin() * 3.0).sin();
        d -= fur.abs() * (0.008 + creature.gene(FIBRES) * 0.016);
    }
    let material = if inside {
        10
    } else if ridge < 0.2 && point.y > 4.0 {
        12
    } else {
        0
    };
    let mut hit = with_wings(Surface::new(d, material), point, creature, LOOK);
    hit = head_surface(point, creature, hit);
    // The tail's fan of feather vanes, spread across the air from its tip, and a smaller pair
    // before it.
    let tip = tail_joint(creature, 7);
    let onward = tip - tail_joint(creature, 6);
    let heading = onward.x.atan2(onward.z);
    for index in 0..7 {
        let spread = (index as f32 - 3.0) * 0.22;
        let angle = heading + spread;
        let direction = vec3(angle.sin(), -0.05, angle.cos());
        let length = 13.0 * (1.0 - spread.abs() * 0.3);
        let root = tail_joint(creature, 6).lerp(tip, 0.5);
        hit.join(
            vane(point, root, direction / direction.length(), length, 1.8),
            1,
        );
    }
    let before = tail_joint(creature, 5);
    for side in 0..2 {
        let sign = if side == 0 { -1.0 } else { 1.0 };
        let direction = vec3(sign * 0.75, -0.05, 0.66);
        hit.join(
            vane(point, before, direction / direction.length(), 6.0, 1.0),
            1,
        );
    }
    // The aur sacs set in the ridge, four to a side, glowing through iron collars thicker the more
    // iron it has gathered: the organs the shield is let out from.
    let sheath = 1.06 + creature.stores.x.min(1.0) * 0.14;
    let size = vec3(0.75, 0.6, 1.05) * (0.8 + creature.gene(AUR_SACS) * 0.4);
    for index in 0..4 {
        let angle = (0.24 + index as f32 * 0.17) * PI;
        let z = RING.y + RING.z * angle.cos();
        let at = mirrored - vec3(RING.x * angle.sin(), ring_height(z) + 0.15, z);
        hit.blend(rounded(at, size), 9, 0.3);
        let collar = rounded(at, size * sheath)
            .max(-rounded(at, size * 0.98))
            .max(-at.y - size.y * 0.1);
        hit.blend(collar, 11, 0.05);
    }
    // The rosette at the middle of the back, where the dome is highest.
    hit.blend(
        ellipsoid(point - vec3(0.0, 5.12, 2.0), vec3(0.9, 0.16, 0.9)),
        9,
        0.2,
    );
    hit
}

/// The head: the eyes set low under the brow, the exchange frills fanning behind them where the
/// air it flies through strikes them first, and the mouth tentacles beneath.
fn head_surface(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    if point.z > -4.0 {
        return hit;
    }
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let eye = (mirrored - on_head(vec3(0.55, -0.12, -0.82), 0.93)).length() - 0.8;
    hit.socket(eye, 0.8);
    let low = (mirrored - on_head(vec3(0.4, -0.85, -0.35), 0.97)).length() - 0.4;
    hit.socket(low, 0.4);
    for i in 0..3 {
        let index = i as f32;
        let root = on_head(vec3(1.0, 0.1 + index * 0.25, 0.3 + index * 0.15), 0.95);
        let tip = root
            + vec3(5.0 - index * 0.8, 1.2 + index * 1.0, 4.5) * (0.8 + creature.gene(FRILLS) * 0.4);
        let axis = (tip - root).normalize();
        let across = axis.cross(Vec3::Z).normalize();
        let normal = axis.cross(across);
        let relative = mirrored - root;
        let along = relative.dot(axis);
        let serration = 1.0 - (along * 5.0).sin().abs() * 0.22;
        let frond = vec3(
            relative.dot(across) / serration,
            relative.dot(normal),
            along - (tip - root).length() * 0.5,
        );
        let membrane = ellipsoid(frond, vec3(0.85, 0.12, (tip - root).length() * 0.55)) * 0.65;
        let stem = tapered_tail(mirrored, root, tip, 0.3, 0.04);
        hit.blend(smooth_min(membrane, stem, 0.1), 2, 0.3);
    }
    hit.blend(
        oral_tentacles(point, creature, MOUTH, REACH, ORAL_PAIRS),
        2,
        0.3,
    );
    hit
}

pub fn render_opaque(ray: Ray, creature: &Tetralate) -> Hit {
    trace_material(ray, creature, surface, material)
}

fn material(point: Vec3, id: u32) -> Material {
    // A pale back over a dark belly, the countershading of a world lit from below, crossed by
    // broad bands of the herd's own green.
    let bands = (point.z * 0.35 + (point.x * 0.5).sin() * 2.0)
        .sin()
        .smoothstep(0.5, 0.85);
    let speckles = ((point.x * 2.3).sin() * (point.z * 3.1).sin()).abs();
    let back = point.y.smoothstep(-1.0, 3.5);
    let colour = vec3(0.04, 0.13, 0.16)
        .lerp(vec3(0.20, 0.47, 0.46), back)
        .lerp(vec3(0.66, 0.70, 0.56), back * bands * 0.5);
    // The pleats of the belly pale in their grooves, where the skin stretches as the crop fills.
    let pleat = (1.0 - (point.x * 1.4).sin().abs()).power(6) * (-1.5 - point.y).clamp(0.0, 1.0);
    let mut m = Material::skin(colour.lerp(vec3(0.62, 0.58, 0.48), pleat * 0.7), 0.45);
    m.texture = WRINKLED;
    m.relief = 0.3;
    m.grain = 0.45;
    m.scatter = 0.5;
    match id {
        1 => {
            // Feather vanes, deep teal shading to gold at the tips.
            let vane = (point.z * 4.0 + point.x.abs() * 7.0).sin() * 0.035;
            let gold = (point.z - 48.0).smoothstep(0.0, 10.0);
            m.color =
                vec3(0.07, 0.28, 0.33).lerp(vec3(0.70, 0.56, 0.26), gold * 0.6) * (1.0 + vane);
            m.roughness = 0.65;
        }
        2 => {
            m.color = vec3(0.79, 0.29, 0.39).lerp(vec3(0.98, 0.72, 0.52), speckles);
            m.translucency = 0.4;
        }
        3 => {
            m.color = vec3(0.008, 0.025, 0.04);
            m.roughness = 0.08;
        }
        9 => {
            m.color = vec3(0.19, 0.82, 0.86);
            m.emission = 0.8;
            m.roughness = 0.18;
        }
        11 => {
            // The iron sheathing the aur sacs.
            m.color = vec3(0.36, 0.37, 0.38);
            m.metal = 0.8;
            m.roughness = 0.35;
        }
        12 => {
            // The ridge: pale and smooth, faintly lit from within along its crest.
            m.color = vec3(0.70, 0.74, 0.66);
            m.roughness = 0.3;
            m.emission = 0.04;
        }
        10 => {
            m.color = vec3(0.07, 0.21, 0.12).lerp(vec3(0.23, 0.36, 0.16), speckles);
            m.roughness = 0.92;
            m.texture = FURRED;
            m.relief = 0.6;
            m.grain = 1.5;
        }
        _ => {}
    }
    m
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[pair(AUR_SAC, vec3(3.9, 4.3, 1.5), vec3(0.6, 0.45, 0.8))];

/// Its broad, feathered wings, banded with gold, on thick stalks: a cruiser's wings, beaten slowly.
pub const LOOK: Look = Look::coat(
    vec4(0.20, 0.47, 0.46, 0.0),
    vec3(0.16, 0.46, 0.46),
    [vec4(34.0, 8.0, -6.0, 0.98), vec4(24.0, 6.0, 7.0, 0.98)],
)
.hinged([vec2(6.8, 1.2), vec2(6.0, 0.8)])
.stalked([vec2(0.6, 0.18), vec2(0.55, 0.18)])
.styled([vec4(0.5, 0.8, 6.0, 0.0), vec4(0.45, 0.8, 5.0, 0.0)])
.flying(vec4(0.2, 0.35, 0.45, 1.3))
.marked(vec4(1.0, 3.0, 0.35, 0.42), vec3(0.72, 0.58, 0.28));

pub fn render(ray: Ray, creature: &Tetralate) -> TraceResult {
    let mut result = TraceResult::opaque(render_opaque(ray, creature));
    visit_thin(ray, creature, |hit| result.add(hit));
    result
}

pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, creature: &Tetralate, mut emit: F) {
    visit_wings(ray, creature, LOOK, &mut emit);
    shield(ray, creature, &mut emit);
}

/// The shield's field at a point: negative inside it. A dome rising from the ridge, as high again
/// as the back is wide, pinned to the ridge at its rim and wavering above it as the charge moves.
fn shield_field(point: Vec3, time: f32, grown: f32) -> f32 {
    let base = ring_height(point.z);
    let span = 0.75 + grown * 0.25;
    let q = vec3(
        point.x / RING.x,
        (point.y - base) / (4.8 * span),
        (point.z - RING.y) / RING.z,
    );
    if q.y < 0.0 {
        return 1.0;
    }
    let angle = q.z.atan2(q.x);
    let rise = (q.y / 0.3).clamp(0.0, 1.0);
    let waver = ((angle * 3.0 + time * 0.35).sin() * 0.045
        + (angle * 5.0 - time * 0.5 + q.y * 4.0).sin() * 0.03)
        * rise;
    q.length() - 1.0 - waver
}

/// The shield over its back, from its twentieth year: a thin, nearly clear dome let out along the
/// ridge from the aur sacs in it, brightest where it leaves the ridge and at a glancing view, with
/// faint bands of light climbing it from the rim.
fn shield<F: FnMut(Hit)>(ray: Ray, creature: &Tetralate, mut emit: F) {
    let grown = shielded(creature);
    if grown <= 0.0 {
        return;
    }
    let local = local_ray(ray, creature);
    let time = creature.animation.z;
    let centre = vec3(0.0, 8.0, RING.y);
    let span = intersect_sphere(local.origin - centre, local.direction, 12.5);
    let start = span.x.max(0.0);
    let end = span.y.min(ray.max_distance);
    if start >= end {
        return;
    }
    let field = |t: f32| shield_field(local.origin + local.direction * t, time, grown);
    let steps = 48;
    let step = (end - start) / steps as f32;
    let mut previous = field(start);
    let mut found = 0;
    for index in 1..=steps {
        if found >= 2 {
            break;
        }
        let t = start + step * index as f32;
        let value = field(t);
        if (previous < 0.0) != (value < 0.0) {
            // Narrowed down by halving.
            let (mut low, mut high) = (t - step, t);
            for _ in 0..8 {
                let middle = f32::midpoint(low, high);
                if (field(low) < 0.0) == (field(middle) < 0.0) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let distance = f32::midpoint(low, high);
            let point = local.origin + local.direction * distance;
            let height = (point.y - ring_height(point.z)) / 4.8;
            // The open rim, where the dome meets the ridge, is no surface.
            if height > 0.02 {
                let epsilon = 0.05;
                let gradient = vec3(
                    shield_field(point + vec3(epsilon, 0.0, 0.0), time, grown)
                        - shield_field(point - vec3(epsilon, 0.0, 0.0), time, grown),
                    shield_field(point + vec3(0.0, epsilon, 0.0), time, grown)
                        - shield_field(point - vec3(0.0, epsilon, 0.0), time, grown),
                    shield_field(point + vec3(0.0, 0.0, epsilon), time, grown)
                        - shield_field(point - vec3(0.0, 0.0, epsilon), time, grown),
                );
                let normal = gradient / gradient.length().max(1e-6);
                let view = local.direction / local.direction.length().max(1e-6);
                let cosine = normal.dot(view).abs();
                let tint = vec3(0.25, 0.68, 0.9);
                let opacity = 0.035 * grown;
                let mut material = Material::skin(tint, 0.18);
                material.ior = 1.36;
                material.transmission =
                    (-(Vec3::ONE - tint) * (opacity / cosine.max(0.08))).exp() * (1.0 - opacity);
                let rim = (1.0 - height.clamp(0.0, 1.0)).power(6);
                let climbing = (height * 18.0 - time * 1.2).sin().max(0.0).power(4)
                    * (1.0 - height.clamp(0.0, 1.0));
                material.emission =
                    (0.0125 + 0.4875 * (1.0 - cosine).power(5) + rim * 0.6 + climbing * 0.08)
                        * grown;
                let mut hit = make_hit(
                    ray,
                    creature,
                    distance,
                    if normal.dot(local.direction) > 0.0 {
                        -normal
                    } else {
                        normal
                    },
                    point,
                    material,
                );
                hit.part = 2;
                emit(hit);
                found += 1;
            }
        }
        previous = value;
    }
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! A Nætu's voice is a call so deep it is felt more than heard, rolling through the great gas
    //! chambers and carrying across a hundred kilometres of open sky. A herd strung out across the
    //! clouds keeps together by it, and the Næ who live on them ask them to call for them.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Situation},
            physiology::Physiology,
            tetralata::{FOLLOW, vivipara::ferrilaminata::naeturidae::LOYALTY},
        },
        behaviour::Goal,
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 24.0,
        voice: 20_000.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        act,
        ..super::super::ETHOLOGY
    };

    /// Making for the herd, a Nætu that has lost sight of it calls, and the herd answers.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        let goal = (super::super::ETHOLOGY.act)(mind, s);
        // The more loyal, the sooner it calls after the herd drawing away.
        let limit = 600.0 - s.temper(LOYALTY) * 400.0;
        let apart = s.seen.kin.is_none_or(|kin| kin.length() > limit);
        if mind.intent == FOLLOW && apart {
            return Some(goal.unwrap_or_else(|| (FOLLOW.act)(mind, s)).calling());
        }
        goal
    }
}
