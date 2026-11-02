use super::{feather_strip, plume, shaft};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::tetralata::vivipara::ferrilaminata::naeturidae::AUR_SAC,
    organs::{Part, pair},
};
use crate::scene::{
    life::{
        Look, Tetralate,
        animalia::{
            tetralata::{oral_grasping, rig_grasping, tentacle_tip},
            visit_wings, with_wings,
        },
        biology::{Biology, Litter},
        genome::Gene,
        louvres, prepare_pose, rounded, smooth_min,
        surface::{Hit, Material, Ray, Surface, TraceResult, WRINKLED, trace_material},
        tapered_tail,
    },
    math::Power,
    planet::GRAVITY,
};
use spirv_std::glam::{FloatExt, Quat, Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 5500.0,
    lifespan_days: 44000.0,
    adult_mass: 0.3,
    // One child, now and then two.
    litter: Litter::new(1, 1, [0.9, 0.09, 0.01, 0.0, 0.0, 0.0]),
    gestation_days: 300.0,
    altitude: (300.0, 12000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 3.2;

/// How long the feathers up its back, how large the aur sacs on its flanks, how plump its body, how
/// long its train, and how full the plumes behind its cheeks.
///
/// A family's back feathers lie alike, so kin know one another at a glance; large sacs heal it
/// faster and light its streamers brighter; a plump body carries it further between meals; a long
/// train wins a mate; and full plumes breathe more of the air, for longer flights between drinks.
pub const CREST: Gene = Gene(38);
pub const SACS: Gene = Gene(39);
pub const HEFT: Gene = Gene(40);
pub const TRAIN: Gene = Gene(41);
pub const PLUMES: Gene = Gene(42);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

/// Four pairs of mouth tentacles, the outer pair grown long for tools: the Næ's hands.
const ORAL_PAIRS: u32 = 4;

taxon!(
    Species,
    "Nae",
    "Næ",
    [
        "Næ are small, intelligent four-winged animals with large round heads, soft plump bodies, small feathers lying up the back from the tail to the crown, and a long tail that droops under a train of forked streamers.",
        "As all their phylum do, they see all round at once, through a circlet of lensed eyes ringing the head; in the Næ it is reduced to three on each side in a rough line, the forward pair much the largest, set close together for the fine work of their tentacles, the middle pair a little higher and the rear a little lower, and the face they make is the first thing a Nætu learns to know.",
        "The mouth opens under the front of the snout, a soft round lip among the roots of the tentacles that bring food and water up to it.",
        "The feathery exchange plumes of their genus have gathered close round the upward-looking pair of eyes, curling back over them like lashes: they shade those eyes from the glare of the polar sunbursts above, and wick the mist that condenses on them across the lenses, keeping them wet in dry air, while still taking breath from the air streaming over the head. A Næ flushes them when it is glad.",
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs. The outer pair is grown long and thick, each ending in three fine, opposable fingers that hold tools, knot fibre, work iron and part a companion's plumes to groom them; the shorter inner pairs steady things and guide food.",
        "Their wings are small, rounded and feathered, on narrow stalks that twist through a quick stroke, for hovering close to work rather than for distance; a Næ travels far on its Nætu.",
        "The bare tip of the tail curls into a hook below the train, and with it a Næ holds fast to the fibres of its Nætu's back through any turn; the train itself is for one another, spread and rippled in greeting and courtship, its streamers lit along their ribs by the aur sacs.",
        "Moisture-recovery folds support short flights, although active travel still requires drinks from living basins.",
        "Isolated aur sacs support their magical abilities and are replenished through the protected life they share with bonded Nætu.",
        "They take iron in their diet as all their order does, and their bodies sheathe their aur sacs in it as their Nætu do; what they gather beyond that they work by hand into rings, threaded on the stems of their plumes, given and traded between friends, so that a Næ wears the record of everyone it has known.",
        "Families groom their companions, tend sheltered homes among dorsal fibres and navigate between drifting communities by currents and the distant polar beams.",
    ],
);

/// The head: its centre and half extents. A Næ is mostly head, as the young of every tetralate
/// are, and keeps that build all its life.
const HEAD: Vec3 = vec3(0.0, 0.03, -0.15);
const HEAD_SIZE: Vec3 = vec3(0.17, 0.15, 0.16);

/// The body behind it, plump round the crop.
const BODY: Vec3 = vec3(0.0, -0.01, 0.07);
const BODY_SIZE: Vec3 = vec3(0.15, 0.14, 0.18);

/// Where the mouth opens, under the front of the snout, and its tentacles with it.
const MOUTH: Vec3 = vec3(0.0, -0.075, -0.235);

/// How far the mouth tentacles reach, as a share of the lineage's usual reach, and how much longer
/// again the outer, working pair grows.
const REACH: f32 = 0.22;
const GRIP: f32 = 1.1;

/// A point on the head's surface, `inset` of the way out along `direction` from its centre.
fn on_head(direction: Vec3, inset: f32) -> Vec3 {
    HEAD + HEAD_SIZE * (direction / direction.length()) * inset
}

/// Eye `index` of the circlet on each side, from the front round to behind: where it lies, how
/// large it is, and which way it looks. They run in a rough line along the head, the forward pair
/// largest and level for close work, the middle pair set a little higher to look a little up and
/// the rear pair a little lower to look a little down, so between them they cover above and below
/// as well as all round.
fn circlet(index: u32) -> (Vec3, f32, Vec3) {
    let angle = 0.3 + index as f32 * 0.48;
    let rise = if index == 0 {
        0.12
    } else if index == 1 {
        0.42
    } else {
        -0.18
    };
    let looking = vec3(angle.sin(), rise, -angle.cos());
    let looking = looking / looking.length();
    let radius = if index == 0 { 0.034 } else { 0.021 };
    (on_head(looking, 0.97), radius, looking)
}

/// How many eyes ring each side of the head.
const CIRCLET: u32 = 3;

/// The downward pair beneath the head, watching the aur.
fn downward() -> (Vec3, f32, Vec3) {
    let looking = vec3(0.35, -0.85, -0.42);
    let looking = looking / looking.length();
    (on_head(looking, 0.97), 0.016, looking)
}

pub fn animate(creature: &mut Tetralate) {
    prepare_pose(creature, 0.4, 0.8, 0);
    creature.opaque_bounds = vec3(0.7, 0.9, 1.5).extend(0.0);
    rig_grasping(creature, MOUTH, REACH, ORAL_PAIRS, GRIP);
    rig_train(creature);
}

/// Knots along the tail and along each strand of the train.
const KNOTS: usize = 6;

/// Strands across the train, which hang from the tail side by side.
const STRANDS: usize = 4;

/// Where the tail leaves the body, and how long each stretch of it is.
const TAIL_ROOT: Vec3 = vec3(0.0, -0.02, 0.2);
const TAIL_STEP: f32 = 0.12;

/// The knot along the tail from which the train hangs, and how long each stretch of a strand is.
const TRAIN_KNOT: usize = 2;
const TRAIN_STEP: f32 = 0.14;

/// How far apart the strands of the train hang at their roots, and how much further apart they
/// spread with each knot towards their ends.
const SPREAD: f32 = 0.045;
const FAN: f32 = 1.7;

/// How far apart strands `a` and `b` of the train are meant to lie at knot `knot`.
fn spacing(knot: usize) -> f32 {
    SPREAD * (1.0 + knot as f32 * FAN)
}

/// Pose the tail and its train by simulating them, as a body moving through the air pulls them.
///
/// The tail is a chain held by its own muscle in a gentle droop behind the body, swinging under
/// gravity and the air it flies through. The train hangs from it as a sheet: four strands rooted
/// side by side on the tail, each a chain of its own and each held to its neighbours knot by knot,
/// so the whole moves as cloth does, billowing and folding rather than as separate sticks. Settled
/// from two starts on a fixed grid and crossfaded, so the pose moves smoothly from frame to frame.
/// The tail goes in the first chain of the rig's ribbons and the strands in the next four.
#[expect(
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "the tail, the strands and the ties across them are stepped together, by knot and \
              strand, as the rest of the shader crate does"
)]
fn rig_train(creature: &mut Tetralate) {
    const STEP: f32 = 1.0 / 120.0;
    const SETTLE: f32 = 2.0;
    let turn = Quat::from_array(creature.rotation.to_array()).conjugate();
    let gravity = turn * vec3(0.0, -GRAVITY, 0.0);
    let flow = vec3(0.0, 0.0, creature.animation.w.max(1.5));
    let end = creature.animation.z.max(0.0);
    let droop = vec3(0.0, -0.22, 1.0) / vec3(0.0, -0.22, 1.0).length();
    let rest_tail = |knot: usize| TAIL_ROOT + droop * TAIL_STEP * knot as f32;
    let lane = |strand: usize| strand as f32 - (STRANDS as f32 - 1.0) * 0.5;
    let rest_strand = |tail: Vec3, strand: usize, knot: usize| {
        let out = vec3(lane(strand) * FAN * 0.35, -0.18, 1.0);
        tail + vec3(lane(strand) * SPREAD, 0.0, 0.0) + out / out.length() * TRAIN_STEP * knot as f32
    };
    let settle = |start: f32| {
        let start = start.max(0.0);
        let mut tail = [Vec3::ZERO; KNOTS];
        for (knot, point) in tail.iter_mut().enumerate() {
            *point = rest_tail(knot);
        }
        let mut train = [[Vec3::ZERO; KNOTS]; STRANDS];
        for (strand, chain) in train.iter_mut().enumerate() {
            for (knot, point) in chain.iter_mut().enumerate() {
                *point = rest_strand(tail[TRAIN_KNOT], strand, knot);
            }
        }
        let (mut tail_before, mut train_before) = (tail, train);
        let steps = ((end - start) / STEP) as usize;
        for step in 0..steps {
            let time = start + step as f32 * STEP;
            let gust = vec3((time * 1.3).sin() * 0.6, (time * 0.9).cos() * 0.3, 0.0);
            // The tail: muscle holds it towards its droop, strongest at the root.
            let last = tail_before;
            tail_before = tail;
            for knot in 1..KNOTS {
                let t = knot as f32 / (KNOTS - 1) as f32;
                let velocity = (tail[knot] - last[knot]) * 0.97;
                let tone = (rest_tail(knot) - tail[knot]) * (90.0 * (1.0 - t * 0.6));
                let drag = (flow + gust - velocity / STEP) * 0.6;
                tail[knot] += velocity + (gravity * 0.4 + drag + tone) * STEP * STEP;
            }
            for _ in 0..2 {
                for knot in 1..KNOTS {
                    let delta = tail[knot] - tail[knot - 1];
                    let fix = delta * (1.0 - TAIL_STEP / delta.length().max(1e-6));
                    if knot == 1 {
                        tail[knot] -= fix;
                    } else {
                        tail[knot - 1] += fix * 0.5;
                        tail[knot] -= fix * 0.5;
                    }
                }
            }
            tail[0] = TAIL_ROOT;
            // The train: passive cloth, but for a little stiffness in each feather's shaft.
            let last = train_before;
            train_before = train;
            for strand in 0..STRANDS {
                for knot in 1..KNOTS {
                    let velocity = (train[strand][knot] - last[strand][knot]) * 0.985;
                    let flutter = vec3(
                        0.0,
                        (time * 4.0 + lane(strand) * 1.3 + knot as f32 * 0.9).sin() * 0.6,
                        0.0,
                    );
                    let drag = (flow + gust + flutter - velocity / STEP) * 1.2;
                    let shape =
                        (rest_strand(tail[TRAIN_KNOT], strand, knot) - train[strand][knot]) * 12.0;
                    train[strand][knot] += velocity + (gravity + drag + shape) * STEP * STEP;
                }
            }
            for _ in 0..3 {
                for strand in 0..STRANDS {
                    for knot in 1..KNOTS {
                        let delta = train[strand][knot] - train[strand][knot - 1];
                        let fix = delta * (1.0 - TRAIN_STEP / delta.length().max(1e-6));
                        if knot == 1 {
                            train[strand][knot] -= fix;
                        } else {
                            train[strand][knot - 1] += fix * 0.5;
                            train[strand][knot] -= fix * 0.5;
                        }
                    }
                }
                // Each strand held to the next across the sheet, as the vanes between them hold.
                for strand in 1..STRANDS {
                    for knot in 1..KNOTS {
                        let delta = train[strand][knot] - train[strand - 1][knot];
                        let fix = delta * (1.0 - spacing(knot) / delta.length().max(1e-6)) * 0.5;
                        train[strand - 1][knot] += fix * 0.5;
                        train[strand][knot] -= fix * 0.5;
                    }
                }
                for strand in 0..STRANDS {
                    train[strand][0] = tail[TRAIN_KNOT] + vec3(lane(strand) * SPREAD, 0.0, 0.0);
                }
            }
        }
        (tail, train)
    };
    let window = (end / SETTLE).floor();
    let weight = end / SETTLE - window;
    let (tail_earlier, train_earlier) = settle((window - 1.0) * SETTLE);
    let (tail_later, train_later) = settle(window * SETTLE);
    for knot in 0..KNOTS {
        creature.rig.ribbons[knot] = tail_earlier[knot]
            .lerp(tail_later[knot], weight)
            .extend(0.0);
        for strand in 0..STRANDS {
            creature.rig.ribbons[(strand + 1) * KNOTS + knot] = train_earlier[strand][knot]
                .lerp(train_later[strand][knot], weight)
                .extend(0.0);
        }
    }
}

/// Knot `knot` of the tail as the simulation left it.
fn tail_knot(creature: &Tetralate, knot: usize) -> Vec3 {
    creature.rig.ribbons[knot].truncate()
}

/// Knot `knot` of the train, `across` of the way from its first strand to its last.
fn train_knot(creature: &Tetralate, across: f32, knot: usize) -> Vec3 {
    let at = across.clamp(0.0, 1.0) * (STRANDS - 1) as f32;
    let strand = (at as usize).min(STRANDS - 2);
    let between = at - strand as f32;
    let a = creature.rig.ribbons[(strand + 1) * KNOTS + knot].truncate();
    let b = creature.rig.ribbons[(strand + 2) * KNOTS + knot].truncate();
    a.lerp(b, between)
}

/// A Næ: small, soft and mostly head.
///
/// Its big round head runs without a waist into a plump body. A circlet of lensed eyes rings the
/// head, the forward pair much the largest, and the mouth opens beneath it among the roots of the
/// tentacles, the outer pair long and thick and ending in fingers: its hands. Three feathery plumes
/// fan back behind each cheek, threaded with rings of worked iron, the aur sacs glow on its flanks,
/// and small feathers lie up its back from the tail to the crown. The tail droops under a train of
/// forked streamers, its bare tip curled into a hook beneath them.
pub fn surface(point: Vec3, creature: &Tetralate) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let heft = 0.94 + creature.gene(HEFT) * 0.12;
    let head = rounded(point - HEAD, HEAD_SIZE);
    let body = rounded(point - BODY, BODY_SIZE * vec3(heft, heft, 1.0));
    let belly = rounded(
        point - vec3(0.0, -0.055, 0.06),
        vec3(0.13, 0.1, 0.15) * heft,
    );
    // Head into body without a waist, as one soft form.
    let mut d = smooth_min(smooth_min(head, body, 0.12), belly, 0.06);
    // A soft rounded snout ahead of the eyes, and full cheeks behind them where the plumes root.
    let snout = rounded(point - vec3(0.0, 0.0, -0.27), vec3(0.08, 0.07, 0.05));
    d = smooth_min(d, snout, 0.06);
    let cheek = rounded(mirrored - vec3(0.11, -0.05, -0.09), vec3(0.06, 0.06, 0.07));
    d = smooth_min(d, cheek, 0.05);
    // Soft pads of brow over the forward eyes, a furrow down the crown between them, a throat
    // pouch behind the mouth that swells as it drinks, and the shoulders the wings root in.
    let brow = rounded(
        mirrored - vec3(0.065, 0.085, -0.255),
        vec3(0.045, 0.022, 0.035),
    );
    d = smooth_min(d, brow, 0.03);
    if point.y > 0.08 && point.z < 0.0 {
        d += (1.0 - (point.x.abs() / 0.025).min(1.0)).power(2) * 0.006;
    }
    let throat = rounded(point - vec3(0.0, -0.11, -0.1), vec3(0.085, 0.06, 0.13));
    d = smooth_min(d, throat, 0.04);
    let shoulder = rounded(mirrored - vec3(0.115, 0.055, 0.02), vec3(0.06, 0.055, 0.07));
    d = smooth_min(d, shoulder, 0.05);
    // Fine pleats along the belly, as all its genus has, that open as the crop fills.
    if point.y < -0.08 && point.z > -0.1 && point.z < 0.2 {
        d += (1.0 - (point.x * 90.0).sin().abs()).power(6) * 0.003;
    }
    // The mouth, a soft round opening under the front of the snout, among the tentacle roots.
    let opening = (point - vec3(0.0, -0.092, -0.24)).length() - 0.017;
    d = d.max(-opening);
    d = louvres(d, point - vec3(0.0, 0.0, 0.08), vec3(0.15, 0.13, 0.18));
    // The tail as the simulation left it, and the bare hook its tip curls into, below the train.
    let tail = shaft(
        point,
        |index| tail_knot(creature, index as usize),
        KNOTS as u32,
        0.06,
        0.02,
    );
    d = smooth_min(d, tail, 0.06);
    let tip = tail_knot(creature, KNOTS - 1);
    let onward = tip - tail_knot(creature, KNOTS - 2);
    let onward = onward / onward.length().max(1e-4);
    let bend = tip + onward * 0.035 + vec3(0.0, -0.035, 0.0);
    let hook = tapered_tail(point, tip, bend, 0.02, 0.016).min(tapered_tail(
        point,
        bend,
        bend + vec3(0.0, -0.01, -0.04),
        0.016,
        0.008,
    ));
    d = smooth_min(d, hook, 0.015);
    let (d, plumage) = feathered(d, point);
    let mut hit = with_wings(Surface::new(d, plumage), point, creature, LOOK);
    // Its lip, a soft rosy ring round the mouth.
    let lip = vec2(
        vec2(point.x, point.z + 0.24).length() - 0.022,
        point.y + 0.082,
    )
    .length()
        - 0.008;
    hit.blend(lip, 17, 0.01);
    // The circlet of eyes round the head, and the downward pair beneath it.
    for index in 0..CIRCLET {
        let (at, radius, _) = circlet(index);
        hit.socket((mirrored - at).length() - radius, radius);
    }
    let (at, radius, _) = downward();
    hit.socket((mirrored - at).length() - radius, radius);
    // An aur sac on each flank, glowing through the iron collar its body laid down round it.
    let sac = 0.028 + creature.gene(SACS) * 0.018;
    let at = mirrored - vec3(0.135, 0.0, 0.07);
    hit.join(at.length() - sac, 9);
    let sheath = sac * (1.08 + creature.stores.x.min(1.0) * 0.2);
    let collar = (at.length() - sheath)
        .max(-(at.length() - sac * 0.98))
        .max(at.x - sac * 0.3);
    hit.join(collar, 11);
    hit = plumes(point, creature, hit);
    hit = train(point, creature, hit);
    hit = hands(point, creature, hit);
    hit
}

/// The mouth tentacles, and on the tip of each of the long outer pair three fine fingers, splayed a
/// little apart, that close on whatever it holds.
fn hands(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    hit.blend(
        oral_grasping(point, creature, MOUTH, REACH, ORAL_PAIRS, GRIP),
        2,
        0.04,
    );
    for index in 0..2 {
        let (tip, before) = tentacle_tip(creature, index);
        if (point - tip.truncate()).length() > 0.12 {
            continue;
        }
        let onward = tip.truncate() - before.truncate();
        let onward = onward / onward.length().max(1e-4);
        let flat = onward.cross(Vec3::Y);
        let across = flat / flat.length().max(1e-4);
        let over = across.cross(onward);
        let thick = (tip.w * 2.2).max(0.006);
        let palm = tip.truncate() + onward * thick;
        hit.blend((point - palm).length() - thick * 1.3, 2, thick);
        for finger in 0..3 {
            let turn = (finger as f32 - 1.0) * 0.7;
            let out = onward + across * turn.sin() * 0.8 + over * (turn.cos() - 0.6) * 0.6;
            let out = out / out.length();
            let end = palm + out * thick * 4.5 - over * thick * 1.2;
            hit.blend(
                tapered_tail(point, palm, end, thick * 0.6, thick * 0.25),
                2,
                thick * 0.5,
            );
        }
    }
    hit
}

/// The three feathery plumes behind each cheek, sweeping back and up, with a ring of worked iron
/// on each stem.
fn plumes(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let (eye, radius, looking) = circlet(1);
    if (mirrored - eye).length() > 0.3 {
        return hit;
    }
    let full = 0.85 + creature.gene(PLUMES) * 0.3;
    // They stir with the wingbeat.
    let stir = (creature.animation.z * 3.0).sin() * 0.01;
    // Rooted in an arc round the back and top of the eye's rim, and sweeping back over it.
    let up = vec3(0.0, 1.0, 0.0);
    let behind = looking.cross(up);
    let behind = behind / behind.length().max(1e-4);
    for index in 0..4 {
        let lane = index as f32 / 3.0;
        let around = 0.4 + lane * 1.5;
        let rim = eye + (up * around.cos() - behind * around.sin()) * radius * 1.5;
        let out = looking * 0.5 + up * (0.7 - lane * 0.5) - behind * (0.4 + lane * 0.4);
        let bend = rim + out * 0.055 * full + vec3(0.0, stir, 0.0);
        let tip =
            bend + (out * 0.5 - behind * 0.7 + vec3(0.0, stir - lane * 0.02, 0.0)) * 0.08 * full;
        hit.join(
            plume(mirrored, rim, bend, tip, 0.008, 0.024 * full, 7.0),
            16,
        );
        if index == 1 {
            // A ring of worked iron round one stem, given by a friend.
            let at = rim.lerp(bend, 0.6);
            let axis = (bend - rim) / (bend - rim).length();
            let q = mirrored - at;
            let along = q.dot(axis);
            let around = (q - axis * along).length();
            hit.join(vec2(around - 0.008, along).length() - 0.003, 7);
        }
    }
    hit
}

/// Where a point lies among the small contour feathers that coat the back: its feather's centre
/// across and along, and how far back through it from the feather's base, from zero to one. They
/// lie in staggered rows like shingles, each tip overlapping the base of the one behind.
fn contour(point: Vec3) -> (f32, f32, f32) {
    let rows = point.z / 0.026;
    let row = rows.floor();
    let shift = if (row as i32).rem_euclid(2) == 0 {
        0.0
    } else {
        0.5
    };
    let columns = point.x / 0.03 + shift;
    let across = columns - columns.floor() - 0.5;
    (across, row, rows - row)
}

/// The body under its plumage: a coat of small contour feathers lying flat back along the skin
/// from the crown to the root of the tail, each one's tip lifting a hair over the base of the
/// next, and a soft ruff of down round the neck where the head meets the body. Returns the body
/// and which of the two covers the point, if either.
fn feathered(body: f32, point: Vec3) -> (f32, u32) {
    if body > 0.02 {
        return (body, 0);
    }
    // Over the back and the upper flanks, from behind the brow to the tail's root.
    let coat = point.y.smoothstep(-0.02, 0.06)
        * (point.z + 0.2).smoothstep(0.0, 0.05)
        * (0.32 - point.z).smoothstep(0.0, 0.06);
    let ruff = (1.0 - ((point.z + 0.02) / 0.035).abs()).max(0.0) * point.y.smoothstep(-0.08, -0.02);
    if coat <= 0.0 && ruff <= 0.0 {
        return (body, 0);
    }
    let (across, _, back) = contour(point);
    // Each feather rounded at its tip, rising from its base under the one before to its tip.
    let outline = (across * 2.0).power(2) + (1.0 - back).power(2) * 0.6;
    let lift = back * (1.0 - outline).max(0.0).sqrt() * 0.0045 * coat;
    // The down: soft, fine tufts.
    let around = point.y.atan2(point.x);
    let fluff =
        ((around * 22.0).sin() * 0.5 + 0.5) * ((point.z * 160.0).sin() * 0.3 + 0.7) * 0.007 * ruff;
    let id = if ruff > 0.35 {
        21
    } else if coat > 0.5 {
        20
    } else {
        0
    };
    (body - lift - fluff, id)
}

/// How many streamers fan across the train.
const STREAMERS: u32 = 7;

/// The train: long streamers fanned across the simulated sheet, each a feather narrow at its quill,
/// widest past its middle and rounded at its tip, with barbs angled back from a pale shaft and
/// beads along the shaft lit by the aur sacs; and the jewellery its owner has hung from them.
fn train(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let middle = train_knot(creature, 0.5, 2);
    if (point - middle).length() > 1.0 {
        return hit;
    }
    let long = 0.85 + creature.gene(TRAIN) * 0.3;
    let mut vanes = 1e6f32;
    let mut ribs = 1e6f32;
    let mut beads = 1e6f32;
    for streamer in 0..STREAMERS {
        let across = streamer as f32 / (STREAMERS - 1) as f32;
        // The middle streamers are the longest.
        let reach = (0.75 + (1.0 - (across - 0.5).abs() * 2.0) * 0.25) * long;
        let last = (((KNOTS - 1) as f32 * reach) as usize).max(2);
        // A feather's width a share of the way along it: a quill at the root, widest beyond the
        // middle, and closing to a rounded tip.
        let width = |knot: usize| {
            let u = knot as f32 / last as f32;
            0.008 + 0.075 * (u * 3.0).min(1.0) * (1.0 - u.power(3))
        };
        for knot in 0..KNOTS - 1 {
            if knot >= last {
                break;
            }
            let a = train_knot(creature, across, knot);
            let b = train_knot(creature, across, knot + 1);
            // The sheet's own normal here, across it from strand to strand.
            let side = train_knot(creature, (across + 0.1).min(1.0), knot)
                - train_knot(creature, (across - 0.1).max(0.0), knot);
            let normal = (b - a).cross(side);
            let normal = normal / normal.length().max(1e-6);
            vanes = vanes.min(feather_strip(
                point,
                a,
                b,
                knot as f32 * TRAIN_STEP,
                (width(knot), width(knot + 1)),
                normal,
                10.0,
            ));
            ribs = ribs.min(tapered_tail(point, a, b, 0.0055, 0.004));
            if knot >= 2 && knot + 1 < last {
                beads = beads.min((point - b).length() - 0.008);
            }
        }
    }
    hit.join(vanes, 1);
    hit.join(ribs, 14);
    hit.join(beads, 13);
    jewellery(point, creature, hit)
}

/// The beads a Næ hangs from its streamers on fine strings: none on the young, more the more iron
/// it has gathered and worked, each where and of what its owner chose.
fn jewellery(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let count = (creature.stores.x.min(1.0) * 4.0 + creature.chance(31) * 1.5) as u32;
    let down = Quat::from_array(creature.rotation.to_array()).conjugate() * vec3(0.0, -1.0, 0.0);
    let mut strings = 1e6f32;
    for index in 0..4 {
        if index >= count {
            break;
        }
        let salt = 40 + index * 7;
        let across = (creature.chance(salt) * STREAMERS as f32).floor() / (STREAMERS - 1) as f32;
        let knot = 2 + (creature.chance(salt + 1) * 3.0) as usize;
        let hang = 0.03 + creature.chance(salt + 2) * 0.05;
        let from = train_knot(creature, across.min(1.0), knot);
        // The strings hang below the train, swinging a little as it moves.
        let swing = (creature.animation.z * 2.0 + index as f32 * 1.7).sin() * 0.25;
        let to = from + (down + vec3(swing, 0.0, 0.3)) * hang;
        strings = strings.min(tapered_tail(point, from, to, 0.0022, 0.0022));
        let size = 0.008 + creature.chance(salt + 3) * 0.006;
        // Most are worked iron; a few are lumps of the bright crust of the mists.
        let material = if creature.chance(salt + 4) < 0.6 {
            7
        } else {
            18
        };
        hit.join((point - to - down * size).length() - size, material);
    }
    hit.join(strings, 19);
    hit
}

pub fn render_opaque(ray: Ray, creature: &Tetralate) -> Hit {
    trace_material(ray, creature, surface, material)
}

#[expect(
    clippy::too_many_lines,
    reason = "every tissue of the body reads best in one match"
)]
fn material(point: Vec3, id: u32) -> Material {
    // A pale back and brow over a deep teal belly, the countershading of a world lit from below.
    let back = point.y.smoothstep(-0.02, 0.09);
    let belly = (-point.y).smoothstep(0.02, 0.12);
    // The snout a warm cream, which makes a face of the large eyes either side of it.
    let muzzle = (-point.z).smoothstep(0.22, 0.3) * (1.0 - (point.x.abs() / 0.06).min(1.0));
    let mut m = Material::skin(
        vec3(0.20, 0.50, 0.54)
            .lerp(vec3(0.42, 0.74, 0.72), back)
            .lerp(vec3(0.05, 0.16, 0.22), belly * 0.85)
            .lerp(vec3(0.94, 0.86, 0.70), muzzle * 0.8),
        0.5,
    );
    m.texture = WRINKLED;
    m.relief = 0.08;
    m.grain = 1.2;
    m.scatter = 0.7;
    m.film = 0.4;
    match id {
        2 => {
            // The mouth tentacles, rosy at the tips.
            m.color = vec3(0.30, 0.52, 0.52).lerp(
                vec3(0.92, 0.62, 0.52),
                (-point.y - 0.1).smoothstep(0.0, 0.25),
            );
            m.roughness = 0.4;
            m.translucency = 0.3;
        }
        1 => {
            // Train feathers: deep teal, shading to bronze towards the tips, with fine barbs.
            let tip = (point.y + 0.3).smoothstep(0.1, -0.5);
            let barbs = (point.x * 260.0 + point.z * 90.0).sin() * 0.04;
            m.color =
                vec3(0.07, 0.30, 0.34).lerp(vec3(0.62, 0.46, 0.20), tip * 0.6) * (1.0 + barbs);
            m.roughness = 0.45;
            m.film = 0.8;
            m.relief = 0.0;
        }
        13 => {
            // The beads along a streamer's rib, lit from within by the aur sacs.
            m.color = vec3(0.30, 0.92, 0.86);
            m.emission = 0.9;
            m.roughness = 0.2;
            m.relief = 0.0;
        }
        14 => {
            // A streamer's raised rib, pale gold.
            m.color = vec3(0.92, 0.80, 0.52);
            m.roughness = 0.35;
            m.relief = 0.0;
        }
        18 => {
            // A bead of the bright crust of the mists, polished and faintly lit.
            m.color = vec3(0.30, 0.78, 0.70);
            m.roughness = 0.12;
            m.emission = 0.15;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        19 => {
            // Fibre string, twisted by hand.
            m.color = vec3(0.70, 0.58, 0.40);
            m.roughness = 0.8;
            m.relief = 0.0;
        }
        20 => {
            // Contour feathers, each darker at its base under the one before and pale at the tip.
            let (across, _, back) = contour(point);
            let edge =
                back.smoothstep(0.6, 0.95) * (1.0 - (across.abs() * 2.0).smoothstep(0.7, 1.0));
            m.color = vec3(0.14, 0.40, 0.44).lerp(vec3(0.58, 0.80, 0.76), edge * 0.7);
            m.roughness = 0.5;
            m.relief = 0.0;
            m.film = 0.6;
        }
        21 => {
            // The ruff of down at the neck, soft and pale.
            m.color = vec3(0.78, 0.86, 0.80);
            m.roughness = 0.85;
            m.relief = 0.0;
            m.scatter = 0.8;
        }
        17 => {
            // The lip round the mouth, soft and rosy.
            m.color = vec3(0.90, 0.55, 0.52);
            m.roughness = 0.35;
            m.translucency = 0.35;
        }
        16 => {
            // The plumes, coral fading to gold at their tips, flushed with blood.
            m.color =
                vec3(0.92, 0.42, 0.42).lerp(vec3(0.98, 0.78, 0.46), point.y.smoothstep(0.0, 0.12));
            m.translucency = 0.5;
            m.roughness = 0.45;
            m.relief = 0.0;
        }
        3 => {
            // A glossy lens over an amber iris round a wide dark pupil, measured across whichever
            // eye this is from the line it looks along.
            let mirrored = vec3(point.x.abs(), point.y, point.z);
            let (mut centre, mut radius, mut looking) = downward();
            let mut nearest = (mirrored - centre).length() - radius;
            for index in 0..CIRCLET {
                let (at, size, towards) = circlet(index);
                let distance = (mirrored - at).length() - size;
                if distance < nearest {
                    nearest = distance;
                    centre = at;
                    radius = size;
                    looking = towards;
                }
            }
            let eye = mirrored - centre;
            let across = (eye - looking * eye.dot(looking)).length() / radius;
            let pupil = across.smoothstep(0.32, 0.4);
            let rim = across.smoothstep(0.82, 0.95);
            m.color = vec3(0.01, 0.015, 0.02)
                .lerp(vec3(0.85, 0.60, 0.18), pupil)
                .lerp(vec3(0.05, 0.12, 0.14), rim);
            m.roughness = 0.04;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        7 => {
            // Worked iron, polished bright by handling.
            m.color = vec3(0.62, 0.62, 0.64);
            m.metal = 0.9;
            m.roughness = 0.18;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        11 => {
            // The iron sheathing the aur sacs.
            m.color = vec3(0.36, 0.37, 0.38);
            m.metal = 0.8;
            m.roughness = 0.35;
        }
        9 => {
            // Aur sacs, glowing with what they hold.
            m.color = vec3(0.18, 0.72, 0.70);
            m.roughness = 0.16;
            m.emission = 0.6;
        }
        _ => {}
    }
    m
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[pair(
    AUR_SAC,
    vec3(0.115, 0.0, 0.07),
    vec3(0.025, 0.025, 0.025),
)];

/// Its four small rounded feathered wings, teal edged with gold, on narrow stalks that twist
/// through a quick, shallow stroke: a hoverer's wings.
pub const LOOK: Look = Look::coat(
    vec4(0.22, 0.50, 0.52, 0.0),
    vec3(0.24, 0.56, 0.58),
    [vec4(0.72, 0.3, -0.02, 0.98), vec4(0.52, 0.24, 0.15, 0.98)],
)
.hinged([vec2(0.12, 0.07), vec2(0.12, 0.04)])
.stalked([vec2(0.22, 0.65), vec2(0.22, 0.65)])
.styled([vec4(0.35, 0.4, 0.0, 0.0), vec4(0.25, 0.4, 0.0, 0.0)])
.framed([0.35, 0.3])
.flying(vec4(3.2, 0.5, 0.35, 1.1))
.marked(vec4(4.0, 1.0, 0.45, 0.45), vec3(0.92, 0.74, 0.40));

pub fn render(ray: Ray, creature: &Tetralate) -> TraceResult {
    let mut result = TraceResult::opaque(render_opaque(ray, creature));
    visit_thin(ray, creature, |hit| result.add(hit));
    result
}

pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, creature: &Tetralate, emit: F) {
    visit_wings(ray, creature, LOOK, emit);
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Næ have no voice, and their tentacles are for their tools. They speak in mist-script:
    //! the moisture-recovery folds that save every breath's water can instead let it out, a fine
    //! plume through the louvres that the aur sacs light from within, shaped by the breath into
    //! glyphs that hang glowing in the air. The crest flushes through its colours as they write,
    //! carrying the tone a voice would. Every word costs water, so in dry air the Næ are sparing
    //! speakers and in the wet heart of a cloud they talk freely; and still air keeps a glyph for a
    //! while after its writer has gone, so a message can be left for whoever comes, until the wind
    //! shreds it. Beyond sight they speak through their Nætu, asking them for the deep calls that
    //! carry across the sky, and they read those calls in turn: each helps the other.

    use crate::scene::life::{
        Kind,
        animalia::{
            SOCIABILITY,
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                display_near,
            },
            physiology::Physiology,
            tetralata::{
                FOLLOW,
                vivipara::{CALL, CHATTINESS, ferrilaminata::naeturidae::LOYALTY, notice as hear},
            },
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 30.0,
        voice: 0.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        court,
        act,
        notice,
        ..super::super::ETHOLOGY
    };

    /// One of its kind near enough to read its glyphs.
    pub const SIGNER: Interest = Interest("signer");

    /// Talking in mist-script with another, hovering face to face a body-length apart.
    pub const CONVERSE: Intent = Intent {
        name: "converse",
        act: |_, s| {
            Goal::new(
                Activity::Sign,
                s.seen
                    .mark(SIGNER)
                    .map(|signer| signer - signer.normalize_or_zero() * 1.5),
                0.4,
            )
            .hovering()
        },
    };

    /// A Næ reads the glyphs of any of its kind in sight, and the calls its Nætu give.
    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        hear(glimpse, seen);
        if glimpse.heard && glimpse.other == Kind::Naetu {
            seen.note(CALL, glimpse.offset);
        }
        if !glimpse.heard && glimpse.other == glimpse.kind && glimpse.offset.length() < 150.0 {
            seen.note(SIGNER, glimpse.offset);
        }
    }

    /// The most talkative of animals when fed and unafraid, and the more so the less the words
    /// cost: with water to spare.
    fn urges(s: &Situation, urges: &mut Urges) {
        (super::super::ETHOLOGY.urges)(s, urges);
        if s.seen.mark(SIGNER).is_some() {
            let content = (1.0 - s.drives.hunger) * (1.0 - s.fear).max(0.0);
            let water = 1.0 - s.drives.thirst;
            urges.add(
                CONVERSE,
                (0.1 + s.temper(SOCIABILITY) * 0.3 + s.temper(CHATTINESS) * 0.3) * content * water,
            );
        }
    }

    /// Parted from its family, a Næ asks its Nætu to call for it, and makes for the answer.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        let goal = (super::super::ETHOLOGY.act)(mind, s);
        // The more loyal, the sooner it calls after the herd drawing away.
        let limit = 450.0 - s.temper(LOYALTY) * 300.0;
        let apart = s.seen.kin.is_none_or(|kin| kin.length() > limit);
        if mind.intent == FOLLOW && apart {
            return Some(goal.unwrap_or_else(|| (FOLLOW.act)(mind, s)).calling());
        }
        goal
    }

    /// Courting in mist-script, close by, with no song to carry it.
    fn court(s: &Situation) -> Goal {
        display_near(s.seen, 1.5, s.seen.mark(CALL))
    }
}
