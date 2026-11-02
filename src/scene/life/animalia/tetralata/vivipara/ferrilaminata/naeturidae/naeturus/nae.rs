use crate::scene::life::surface::WRINKLED;

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

/// How long the feathers of its forehead tuft, how large the aur sacs on its flanks, how plump its
/// body, and how long its train.
///
/// A family's tufts rise alike, so kin know one another at a glance; large sacs heal it faster
/// and light its streamers brighter; a plump body carries it further between meals; and a long
/// train wins a mate.
pub const CREST: Gene = Gene(38);
pub const SACS: Gene = Gene(39);
pub const HEFT: Gene = Gene(40);
pub const TRAIN: Gene = Gene(41);

/// Its colouring: the hue of its skin, round the whole wheel; how far its feathers are turned from
/// that; and how boldly it is barred across the back.
///
/// None is better than another for the body; they are how kin know kin. Families share them, and
/// so the Næ of one drifting community, long bred among themselves, come to look alike and unlike
/// those of the next, as different as teal from rose or gold.
pub const TINT: Gene = Gene(43);
pub const SPLASH: Gene = Gene(44);
pub const BARS: Gene = Gene(45);

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
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs. The outer pair is grown long and thick, each ending in three fine, opposable fingers that hold tools, knot fibre, work iron and part a companion's plumes to groom them; the shorter inner pairs steady things and guide food.",
        "Their wings are small, rounded and feathered, on narrow stalks that twist through a quick stroke, for hovering close to work rather than for distance; a Næ travels far on its Nætu.",
        "The bare tip of the tail curls into a hook below the train, and with it a Næ holds fast to the fibres of its Nætu's back through any turn; the train itself is for one another, spread and rippled in greeting and courtship, its streamers lit along their ribs by the aur sacs.",
        "Moisture-recovery folds support short flights, although active travel still requires drinks from living basins.",
        "Isolated aur sacs support their magical abilities and are replenished through the protected life they share with bonded Nætu.",
        "They take iron in their diet as all their order does, and their bodies sheathe their aur sacs in it as their Nætu do; what they gather beyond that they work by hand into gems and beads, given and traded between friends and hung on long strings from the train, so that an old Næ wears the record of everyone it has known.",
        "Their colouring runs in families, every hue of skin and feather among them, so the Næ of one community, long bred among themselves, look alike and unlike those of the next. The sexes differ too: the one that carries the young is heavier, with larger aur sacs, and the other courts with a longer train and a longer tuft and wears the brighter colours.",
        "Families groom their companions, tend sheltered homes among dorsal fibres and navigate between drifting communities by currents and the distant polar beams.",
    ],
);

/// The head: its centre and half extents. A Næ is mostly head, as the young of every tetralate
/// are, and keeps that build all its life.
const HEAD: Vec3 = vec3(0.0, 0.03, -0.15);
const HEAD_SIZE: Vec3 = vec3(0.165, 0.14, 0.185);

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
    rig_train(creature, TRAIN_SHAPE);
}

/// The Næ's tail and train: rooted at the back of its body, at its own small scale.
const TRAIN_SHAPE: Train = Train {
    root: vec3(0.0, -0.02, 0.2),
    scale: 1.0,
};

/// A Næ: small, soft and mostly head.
///
/// Its big round head runs without a waist into a plump body. A circlet of lensed eyes rings the
/// head, the forward pair much the largest, and the mouth opens beneath it among the roots of the
/// tentacles, the outer pair long and thick and ending in fingers: its hands. the aur sacs glow on
/// its flanks, and small feathers lie up its back from the tail to the crown. The tail droops under
/// a train of forked streamers, its bare tip curled into a hook beneath them.
#[expect(
    clippy::too_many_lines,
    reason = "the body's forms are built up one on another and read best in order"
)]
pub fn surface(point: Vec3, creature: &Tetralate) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    // The sex that carries the young is the heavier.
    let heft = 0.94
        + creature.gene(HEFT) * 0.12
        + if creature.sex.x > 0.5 && creature.sex.x < 1.5 {
            0.06
        } else {
            0.0
        };
    let head = rounded(point - HEAD, HEAD_SIZE);
    let body = rounded(point - BODY, BODY_SIZE * vec3(heft, heft, 1.0));
    let belly = rounded(
        point - vec3(0.0, -0.055, 0.06),
        vec3(0.13, 0.1, 0.15) * heft,
    );
    // Head into body without a waist, its crown flattened.
    let head = -smooth_min(-head, -(point.y - 0.155), 0.03);
    let mut d = smooth_min(smooth_min(head, body, 0.12), belly, 0.06);
    d = smooth_min(d, face(point), 0.012);
    // A long rump behind the body, tapering into the tail, so the one runs into the other.
    let rump = rounded(
        point - vec3(0.0, -0.015, 0.21),
        vec3(0.1, 0.085, 0.14) * heft,
    );
    d = smooth_min(d, rump, 0.06);
    // Full cheeks behind the eyes, and a throat pouch behind the mouth that
    // swells as it drinks.
    let cheek = rounded(mirrored - vec3(0.11, -0.05, -0.09), vec3(0.06, 0.06, 0.07));
    d = smooth_min(d, cheek, 0.05);
    let throat = rounded(point - vec3(0.0, -0.11, -0.1), vec3(0.085, 0.06, 0.13));
    d = smooth_min(d, throat, 0.04);
    // The body reaches out to each pair of wings: broad shoulders drawn out into the fore wings'
    // roots and narrower haunches into the hind, so it narrows between them to a waist.
    let shoulder = rounded(mirrored - vec3(0.1, 0.05, 0.0), vec3(0.06, 0.05, 0.07));
    let fore = tapered_tail(
        mirrored,
        vec3(0.05, 0.05, -0.01),
        vec3(0.172, 0.068, -0.02),
        0.055,
        0.03,
    );
    let haunch = rounded(mirrored - vec3(0.09, 0.03, 0.15), vec3(0.05, 0.045, 0.055));
    let hind = tapered_tail(
        mirrored,
        vec3(0.05, 0.03, 0.15),
        vec3(0.152, 0.045, 0.15),
        0.045,
        0.026,
    );
    d = smooth_min(d, shoulder.min(fore), 0.025);
    d = smooth_min(d, haunch.min(hind), 0.025);
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
        0.075,
        0.02,
    );
    d = smooth_min(d, tail, 0.08);
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
    let (d, plumage) = feathered(d, point, creature);
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
    // The sex that carries the young carries larger aur sacs, to heal and shield its brood.
    let bearer = if creature.sex.x > 0.5 && creature.sex.x < 1.5 {
        1.2
    } else {
        1.0
    };
    let sac = (0.028 + creature.gene(SACS) * 0.018) * bearer;
    let at = mirrored - vec3(0.135, 0.0, 0.07);
    hit.join(at.length() - sac, 9);
    let sheath = sac * (1.08 + creature.stores.x.min(1.0) * 0.2);
    let collar = (at.length() - sheath)
        .max(-(at.length() - sac * 0.98))
        .max(at.x - sac * 0.3);
    hit.join(collar, 11);
    hit = tuft(point, creature, hit);
    hit = train(point, creature, hit);
    hit = hands(point, creature, hit);
    hit
}

/// The face's own forms, so the head reads as more than a ball: a squared snout and a small chin, a
/// heavy ridge of brow over each forward eye, a cheekbone running back below it, and a pair of
/// low ridges sweeping from the brows back over the crown into the line of the back.
fn face(point: Vec3) -> f32 {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let snout = rounded(point - vec3(0.0, -0.005, -0.31), vec3(0.068, 0.055, 0.06));
    let chin = rounded(point - vec3(0.0, -0.075, -0.28), vec3(0.05, 0.03, 0.045));
    // The bridge of the snout, a ridge from its tip up between the forward eyes.
    let bridge = tapered_tail(
        point,
        vec3(0.0, 0.035, -0.36),
        vec3(0.0, 0.125, -0.24),
        0.016,
        0.012,
    );
    // A heavy ridge of brow over each forward eye, running on as a ledge along the line of the eyes
    // behind it.
    let brow = tapered_tail(
        mirrored,
        vec3(0.02, 0.105, -0.315),
        vec3(0.075, 0.1, -0.3),
        0.02,
        0.014,
    );
    let cheekbone = tapered_tail(
        mirrored,
        vec3(0.085, -0.035, -0.27),
        vec3(0.16, -0.02, -0.12),
        0.02,
        0.01,
    );
    let crown = tapered_tail(
        mirrored,
        vec3(0.045, 0.105, -0.27),
        vec3(0.035, 0.16, -0.12),
        0.016,
        0.013,
    )
    .min(tapered_tail(
        mirrored,
        vec3(0.035, 0.16, -0.12),
        vec3(0.022, 0.14, 0.02),
        0.013,
        0.006,
    ));
    let mut d = smooth_min(snout, chin, 0.015);
    d = smooth_min(d, bridge, 0.012);
    d = smooth_min(d, brow, 0.01);
    d = smooth_min(d, cheekbone, 0.012);
    smooth_min(d, crown, 0.01)
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

/// The tuft on its forehead: three long feathers rising from between the brows and sweeping back
/// over the crown, the middle one longest, as long as its genes have them and longer in the sex
/// that courts.
fn tuft(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    if (point - vec3(0.0, 0.25, -0.12)).length() > 0.3 {
        return hit;
    }
    let long = (0.85 + creature.gene(CREST) * 0.3) * if creature.sex.x > 1.5 { 1.2 } else { 1.0 };
    let stir = (creature.animation.z * 2.5).sin() * 0.01;
    let mut feathers = 1e6f32;
    for index in 0..3 {
        let lane = index as f32 - 1.0;
        let length = (0.24 - lane.abs() * 0.05) * long;
        let root = vec3(lane * 0.018, 0.14, -0.215);
        let rise = vec3(lane * 0.35, 0.85, 0.45);
        let bend = root + rise / rise.length() * length * 0.5;
        let back = vec3(lane * 0.45, 0.25 + stir * 10.0, 1.0);
        let tip = bend + back / back.length() * length * 0.5;
        for half in 0..2 {
            let (a, b) = if half == 0 { (root, bend) } else { (bend, tip) };
            let normal = (b - a).cross(Vec3::Y);
            let normal = normal / normal.length().max(1e-5);
            let width = |at: f32| 0.004 + 0.016 * (at * 3.0).min(1.0) * (1.0 - at.power(3));
            let start = half as f32 * 0.5;
            feathers = feathers.min(feather_strip(
                point,
                a,
                b,
                start * length,
                (width(start), width(start + 0.5)),
                normal,
                10.0,
            ));
        }
    }
    hit.join(feathers, 1);
    hit
}

/// How large the contour feathers are at a point along the body, a share of their size at the
/// root of the tail: small and fine on the head, growing down the back.
fn plume_scale(z: f32) -> f32 {
    (0.55 + (z + 0.25) * 1.6).clamp(0.5, 1.4)
}

/// A white-noise value in [0, 1) for a feather's cell, so that no two lie quite alike.
fn jitter(cell: Vec2) -> f32 {
    ((cell.x * 12.9898 + cell.y * 78.233).sin() * 43_758.547)
        .fract()
        .abs()
}

/// Where a point lies among the small contour feathers that coat the back: how far across its
/// feather from the middle, how far back through it from base to tip, and how large it is. They
/// lie in staggered rows like shingles, each tip overlapping the base of the one behind, growing
/// from the head down the back, and each a little out of line with its neighbours.
fn contour(point: Vec3) -> (f32, f32, f32, f32) {
    let size = plume_scale(point.z);
    let rows = point.z / (0.026 * size);
    let row = rows.floor();
    let shift = if (row as i32).rem_euclid(2) == 0 {
        0.0
    } else {
        0.5
    };
    let columns = point.x / (0.03 * size) + shift;
    let column = columns.floor();
    let wobble = jitter(vec2(row, column));
    let across = columns - column - 0.5 + (wobble - 0.5) * 0.25;
    let back = (rows - row + (wobble - 0.5) * 0.3).clamp(0.0, 1.0);
    (across, back, size, wobble)
}

/// How far up round the tail's line a point near it lies, from -1 beneath to 1 on top, or -1 if
/// it is nowhere near the tail, so the coat can run on down the top of the tail into the train.
fn tail_top(point: Vec3, creature: &Tetralate) -> f32 {
    let mut best = 1e6f32;
    let mut height = -1.0f32;
    for knot in 0..KNOTS - 1 {
        let a = tail_knot(creature, knot);
        let b = tail_knot(creature, knot + 1);
        let axis = b - a;
        let t = ((point - a).dot(axis) / axis.length_squared().max(1e-6)).clamp(0.0, 1.0);
        let near = a + axis * t;
        let distance = (point - near).length();
        if distance < best {
            best = distance;
            height = (point.y - near.y) / distance.max(1e-4);
        }
    }
    if best > 0.08 { -1.0 } else { height }
}

/// The body under its plumage: a coat of small contour feathers lying flat back along the skin
/// from the crown down the back and on along the top of the tail into the train, each one's tip
/// lifting a hair over the base of the next, fading in softly at its edges. Returns the body and
/// whether a feather covers the point.
fn feathered(body: f32, point: Vec3, creature: &Tetralate) -> (f32, u32) {
    if body > 0.02 {
        return (body, 0);
    }
    // Over the back and the upper flanks from behind the brow, and on down the tail's top.
    let back = point.y.smoothstep(-0.09, 0.07) * (point.z + 0.25).smoothstep(0.0, 0.16);
    let tail = if point.z > 0.15 {
        tail_top(point, creature).smoothstep(-0.2, 0.5)
    } else {
        0.0
    };
    let coat = back.max(tail);
    if coat <= 0.0 {
        return (body, 0);
    }
    let (across, along, size, wobble) = contour(point);
    // Towards the coat's edges the feathers thin out one by one rather than stopping on a line:
    // each grows only where the coat is thicker than its own chance.
    let threshold = 0.1 + wobble * 0.75;
    let present = coat.smoothstep(threshold - 0.06, threshold + 0.06);
    // Each feather rounded at its tip, rising from its base under the one before to its tip.
    let outline = (across * 2.0).power(2) + (1.0 - along).power(2) * 0.6;
    let lift = along * (1.0 - outline).max(0.0).sqrt() * 0.004 * size * present;
    let id = if present > 0.5 { 20 } else { 0 };
    (body - lift, id)
}

/// The train, with the jewellery its owner has hung from it.
fn train(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    // The sex that does not carry the young courts with its train, and grows it longer.
    let long = (0.85 + creature.gene(TRAIN) * 0.3) * if creature.sex.x > 1.5 { 1.25 } else { 0.95 };
    let parts = plumage(point, creature, TRAIN_SHAPE, long);
    hit.join(parts.x, 1);
    hit.join(parts.y, 14);
    hit.join(parts.z, 13);
    jewellery(point, creature, hit)
}

/// The gems a Næ hangs from its train on long strings, gathered over its life: none on the young,
/// more and more as it ages, each where, how far down and of what its owner chose, so no two Næ
/// wear the same. Some strings carry two or three, one below another; each is a round bead, a
/// teardrop or a cut stone, of worked iron, the bright crust of the mists, amber or rose.
fn jewellery(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let worn = ((creature.life.x / BIOLOGY.maturity_days - 0.6) / 2.5).clamp(0.0, 1.0);
    let count = (worn * 8.0 * (0.45 + creature.chance(31) * 0.55)) as u32;
    if count == 0 {
        return hit;
    }
    let down = Quat::from_array(creature.rotation.to_array()).conjugate() * vec3(0.0, -1.0, 0.0);
    let mut strings = 1e6f32;
    for index in 0..8 {
        if index >= count {
            break;
        }
        let salt = 40 + index * 11;
        let (root, across, reach) =
            train_feather((creature.chance(salt) * TRAIN_FEATHERS as f32) as u32);
        let knot = 1 + (creature.chance(salt + 1) * 2.0) as usize;
        let base = tail_knot(creature, root);
        let from =
            base + (train_knot(creature, across, knot) - train_knot(creature, across, 0)) * reach;
        // Long strings, swinging a little as the train moves.
        let hang = 0.1 + creature.chance(salt + 2) * 0.22;
        let swing = (creature.animation.z * 1.6 + index as f32 * 1.7).sin() * 0.15;
        let line = down + vec3(swing, 0.0, 0.2);
        let line = line / line.length();
        let to = from + line * hang;
        if (point - from.lerp(to, 0.5)).length() > hang * 0.6 + 0.06 {
            continue;
        }
        strings = strings.min(tapered_tail(point, from, to, 0.0018, 0.0018));
        let stones = 1 + (creature.chance(salt + 3) * 2.99) as u32;
        for stone in 0..3 {
            if stone >= stones {
                break;
            }
            let salt = salt + 4 + stone * 3;
            let size = 0.007 + creature.chance(salt) * 0.009;
            let centre = to - line * (stone as f32 * hang * 0.28) + line * size;
            let q = point - centre;
            let shape = creature.chance(salt + 1);
            let gem = if shape < 0.34 {
                q.length() - size
            } else if shape < 0.67 {
                // A teardrop, hanging point up.
                let along = q.dot(line);
                let girth = size * (0.4 + (along / size * 0.5 + 0.5).clamp(0.0, 1.0) * 0.6);
                ((q - line * along).length() - girth).max(along.abs() - size * 1.6)
            } else {
                // A cut stone, faceted.
                (q.x.abs() + q.y.abs() + q.z.abs() - size * 1.4) * 0.577
            };
            let pick = creature.chance(salt + 2);
            let material = if pick < 0.35 {
                7
            } else if pick < 0.6 {
                18
            } else if pick < 0.8 {
                22
            } else {
                23
            };
            hit.join(gem, material);
        }
    }
    hit.join(strings, 19);
    hit
}

pub fn render_opaque(ray: Ray, creature: &Tetralate) -> Hit {
    trace_material(ray, creature, surface, move |point, id| {
        material(point, id, creature)
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "every tissue of the body reads best in one match"
)]
fn material(point: Vec3, id: u32, creature: &Tetralate) -> Material {
    // A pale back and brow over a deep teal belly, the countershading of a world lit from below.
    let back = point.y.smoothstep(-0.02, 0.09);
    let belly = (-point.y).smoothstep(0.02, 0.12);
    // The snout a warm cream, which makes a face of the large eyes either side of it.
    let muzzle = (-point.z).smoothstep(0.22, 0.3) * (1.0 - (point.x.abs() / 0.06).min(1.0));
    // Bars across the back, as bold as its genes have them.
    let barred = ((point.z * 34.0 + (point.x * 20.0).sin() * 0.6).sin() * 0.5 + 0.5)
        .smoothstep(0.55, 0.8)
        * back
        * creature.gene(BARS);
    let mut m = Material::skin(
        vec3(0.20, 0.50, 0.54)
            .lerp(vec3(0.42, 0.74, 0.72), back * (1.0 - barred * 0.7))
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
        22 => {
            // Amber, warm and clear.
            m.color = vec3(0.95, 0.58, 0.16);
            m.roughness = 0.1;
            m.translucency = 0.6;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        23 => {
            // Rose stone, polished.
            m.color = vec3(0.86, 0.42, 0.52);
            m.roughness = 0.12;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        19 => {
            // Fibre string, twisted by hand.
            m.color = vec3(0.70, 0.58, 0.40);
            m.roughness = 0.8;
            m.relief = 0.0;
        }
        17 => {
            // The lip round the mouth, soft and rosy.
            m.color = vec3(0.90, 0.55, 0.52);
            m.roughness = 0.35;
            m.translucency = 0.35;
        }
        20 => {
            // Contour feathers, each darker at its base under the one before and pale at the tip.
            let (across, back, _, _) = contour(point);
            let edge =
                back.smoothstep(0.6, 0.95) * (1.0 - (across.abs() * 2.0).smoothstep(0.7, 1.0));
            m.color = vec3(0.14, 0.40, 0.44).lerp(vec3(0.58, 0.80, 0.76), edge * 0.7);
            m.roughness = 0.5;
            m.relief = 0.0;
            m.film = 0.6;
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
    // Its family's colours: the skin turned round the wheel as its genes have it, and its
    // feathers and plumes turned further, so a whole community can be rose, gold or violet where
    // another is teal. Eyes, gems, iron and the glow of the aur keep their own.
    let skin = (creature.gene(TINT) - 0.5) * TAU * 0.9;
    let feathers = skin + (creature.gene(SPLASH) - 0.5) * TAU * 0.6;
    if id == 0 || id == 2 || id == 17 || id == 20 {
        m.color = turned(m.color, skin);
    } else if id == 1 || id == 14 {
        m.color = turned(m.color, feathers);
    }
    m
}

/// A colour turned round the colour wheel by `angle`, keeping how light it is.
fn turned(colour: Vec3, angle: f32) -> Vec3 {
    let axis = Vec3::splat(0.577_350_26);
    let (sine, cosine) = angle.sin_cos();
    (colour * cosine + axis.cross(colour) * sine + axis * axis.dot(colour) * (1.0 - cosine))
        .max(Vec3::ZERO)
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
.hinged([vec2(0.175, 0.068), vec2(0.155, 0.045)])
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
    //! carry across the sky, and they read those calls in turn: each helps the other. A Næ sleeps
    //! at home: on the back of its Nætu, among the dorsal fibres, if one is near.

    use super::*;
    use crate::scene::life::{
        animalia::{
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                display_near,
            },
            physiology::Physiology,
            tetralata::vivipara::notice as hear,
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
        sleep,
        act,
        notice,
        ..super::super::ETHOLOGY
    };

    /// A Nætu, home to the Næ that live on it.
    const CARRIER: Interest = Interest("carrier");

    /// Asleep on a Nætu's back if one is near, or as its lineage sleeps if not.
    fn sleep(s: &Situation) -> Goal {
        s.seen.mark(CARRIER).map_or_else(
            || (super::super::ETHOLOGY.sleep)(s),
            |carrier| {
                Goal::new(Activity::Sleep, Some(carrier + s.up() * 6.0), 0.4)
                    .folded(0.9)
                    .hovering()
            },
        )
    }

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
        if !glimpse.heard && glimpse.other == Kind::Naetu {
            seen.note(CARRIER, glimpse.offset);
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
