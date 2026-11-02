use crate::scene::life::Tetralate;
pub mod nae;
pub mod naetu;

taxon!(
    Genus,
    "Naeturus",
    "Nae Kin",
    [
        "Members of Naeturus share four wings on short stalks, long tails trailing a train of forked streamers lit along their ribs by the aur sacs and aur-storage organs isolated from the digestive circulation.",
        "The genus has diverged into small dexterous Næ and enormous carrying Nætu.",
        "Mouth tentacles retain their ancestral feeding role while also supporting grooming, construction and close social contact.",
        "Their complementary sizes let one lineage maintain dwellings and tools while the other carries families between distant communities.",
        "Shared recognition and dependence on safely stored aur sustain the association through journeys that either form would struggle to make alone.",
    ],
);

/// A tail of the genus: a shaft tapering from `start` to `end` thick along the joints `joint`
/// gives, `count` of them, each running smoothly into the next so it bends as one.
pub fn shaft<F: Fn(u32) -> Vec3>(point: Vec3, joint: F, count: u32, start: f32, end: f32) -> f32 {
    let mut distance = 1e6f32;
    for index in 0..count - 1 {
        let from = index as f32 / (count - 1) as f32;
        let to = (index + 1) as f32 / (count - 1) as f32;
        let a = start + (end - start) * from;
        let b = start + (end - start) * to;
        distance = smooth_min(
            distance,
            tapered_tail(point, joint(index), joint(index + 1), a, b),
            a * 0.5,
        );
    }
    distance
}

/// A flat strip from `a` to `b`, tapering from `start` to `end` wide, lying across `normal` (of
/// unit length) and `flatness` times thinner than it is wide: one stretch of a streamer whose shape
/// a simulation sets.
pub fn strip(
    point: Vec3,
    a: Vec3,
    b: Vec3,
    start: f32,
    end: f32,
    normal: Vec3,
    flatness: f32,
) -> f32 {
    let q = point - a;
    let across = q.dot(normal);
    let squashed = a + (q - normal * across) + normal * across * flatness;
    tapered_tail(squashed, a, b, start, end) / flatness
}

/// One stretch of a feather drawn as a strip from `a` to `b` (see [`strip`]).
///
/// It has the detail that makes it read as a feather: barbs angled back from the shaft down the
/// middle, raised a
/// hair either side of the grooves between them, and an edge broken into fine barb tips. `start`
/// is how far along the whole feather `a` lies, so the barbs run on unbroken from one stretch to
/// the next.
pub fn feather_strip(
    point: Vec3,
    a: Vec3,
    b: Vec3,
    start: f32,
    widths: (f32, f32),
    normal: Vec3,
    flatness: f32,
) -> f32 {
    let axis = b - a;
    let span = axis.length().max(1e-5);
    let direction = axis / span;
    let side = normal.cross(direction);
    let q = point - a;
    let along = q.dot(direction).clamp(0.0, span);
    let across = q.dot(side);
    let width = widths.0 + (widths.1 - widths.0) * (along / span);
    // The edge breaks into the tips of the barbs.
    let ragged = 1.0
        - ((start + along) / width.max(1e-4) * 3.0 + across.abs() * 9.0)
            .sin()
            .abs()
            * 0.1;
    let flat = strip(
        point,
        a,
        b,
        widths.0 * ragged,
        widths.1 * ragged,
        normal,
        flatness,
    );
    // Barbs angled back from the shaft, each a faint ridge.
    let barbs = ((start + along - across.abs() * 0.8) / width.max(1e-4) * 9.0).sin();
    flat - barbs * width / flatness * 0.25
}

/// Knots along the tail and along each strand of the train.
pub const KNOTS: usize = 6;

/// Strands across the train, which hang from the tail side by side.
const STRANDS: usize = 4;

/// The knot along the tail from which the train hangs.
const TRAIN_KNOT: usize = 2;

/// The shape of one animal's tail and train: where the tail leaves its body, and its scale, every
/// length of the tail, the train and their feathers being a multiple of a Næ's.
#[derive(Clone, Copy)]
pub struct Train {
    pub root: Vec3,
    pub scale: f32,
}

impl Train {
    /// How long each stretch of the tail is, and each stretch of a strand of the train.
    const fn tail_step(self) -> f32 {
        0.12 * self.scale
    }

    const fn train_step(self) -> f32 {
        0.14 * self.scale
    }

    /// How far apart the strands of the train hang at their roots, and how much further apart they
    /// spread with each knot towards their ends.
    const fn spread(self) -> f32 {
        0.045 * self.scale
    }

    const FAN: f32 = 1.7;

    /// How far apart neighbouring strands are meant to lie at knot `knot`.
    fn spacing(self, knot: usize) -> f32 {
        self.spread() * (1.0 + knot as f32 * Self::FAN)
    }
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
    clippy::needless_range_loop,
    reason = "the tail, the strands and the ties across them are stepped together, by knot and \
              strand, as the rest of the shader crate does"
)]
pub fn rig_train(creature: &mut Tetralate, shape: Train) {
    const STEP: f32 = 1.0 / 120.0;
    let (gravity, flow) = air_frame(creature, 1.5);
    let end = creature.animation.z.max(0.0);
    let droop = vec3(0.0, -0.22, 1.0) / vec3(0.0, -0.22, 1.0).length();
    let (tail_step, train_step, spread) = (shape.tail_step(), shape.train_step(), shape.spread());
    let rest_tail = |knot: usize| shape.root + droop * tail_step * knot as f32;
    let lane = |strand: usize| strand as f32 - (STRANDS as f32 - 1.0) * 0.5;
    let rest_strand = |tail: Vec3, strand: usize, knot: usize| {
        let out = vec3(lane(strand) * Train::FAN * 0.35, -0.18, 1.0);
        tail + vec3(lane(strand) * spread, 0.0, 0.0) + out / out.length() * train_step * knot as f32
    };
    let settle = |start: f32| {
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
        for step in 0..((end - start) / STEP) as usize {
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
                swing(&mut tail[knot], velocity, gravity * 0.4 + drag + tone, STEP);
            }
            tie(&mut tail, tail_step);
            tie(&mut tail, tail_step);
            tail[0] = shape.root;
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
                    let pull =
                        (rest_strand(tail[TRAIN_KNOT], strand, knot) - train[strand][knot]) * 12.0;
                    swing(
                        &mut train[strand][knot],
                        velocity,
                        gravity + drag + pull,
                        STEP,
                    );
                }
            }
            for _ in 0..3 {
                for chain in &mut train {
                    tie(chain, train_step);
                }
                // Each strand held to the next across the sheet, as the vanes between them hold.
                for strand in 1..STRANDS {
                    for knot in 1..KNOTS {
                        let delta = train[strand][knot] - train[strand - 1][knot];
                        let fix =
                            delta * (1.0 - shape.spacing(knot) / delta.length().max(1e-6)) * 0.5;
                        train[strand - 1][knot] += fix * 0.5;
                        train[strand][knot] -= fix * 0.5;
                    }
                }
                for strand in 0..STRANDS {
                    train[strand][0] = tail[TRAIN_KNOT] + vec3(lane(strand) * spread, 0.0, 0.0);
                }
            }
        }
        (tail, train)
    };
    let ((tail_earlier, train_earlier), (tail_later, train_later), weight) =
        settled(end, 2.0, settle);
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

/// Knot `knot` of the tail as [`rig_train`] left it.
pub fn tail_knot(creature: &Tetralate, knot: usize) -> Vec3 {
    creature.rig.ribbons[knot].truncate()
}

/// Knot `knot` of the train, `across` of the way from its first strand to its last.
pub fn train_knot(creature: &Tetralate, across: f32, knot: usize) -> Vec3 {
    let at = across.clamp(0.0, 1.0) * (STRANDS - 1) as f32;
    let strand = (at as usize).min(STRANDS - 2);
    let between = at - strand as f32;
    let a = creature.rig.ribbons[(strand + 1) * KNOTS + knot].truncate();
    let b = creature.rig.ribbons[(strand + 2) * KNOTS + knot].truncate();
    a.lerp(b, between)
}

/// How many feathers make the train: five rows of four.
pub const TRAIN_FEATHERS: u32 = 20;

/// Train feather `index`: how far along the tail it roots, in tail knots, the first row up on the
/// back before the tail begins; where across the sheet it lies; and how long it is as a share of
/// the longest.
///
/// Five rows, each staggered to fall between the feathers of the row before, growing from short
/// coverts on the back to the long feathers at the tail's end.
pub fn train_feather(index: u32) -> (f32, f32, f32) {
    let row = index / 4;
    let slot = index % 4;
    let stagger = if row.is_multiple_of(2) { -0.2 } else { 0.2 };
    let across = ((slot as f32 + 0.5 + stagger) / 4.0).clamp(0.0, 1.0);
    (row as f32 - 1.0, across, 0.3 + row as f32 * 0.175)
}

/// Where a train feather `along` the tail and `across` the sheet is rooted: on the tail's side or
/// top as its place across has it, never inside it, or up on the back for the coverts before the
/// tail begins.
pub fn feather_root(creature: &Tetralate, shape: Train, along: f32, across: f32) -> Vec3 {
    let side = (across - 0.5) * 2.0;
    let knot = along.max(0.0) as usize;
    let thick = (0.075 - 0.011 * knot as f32) * shape.scale;
    let out = vec3(side * thick * 0.95, (1.0 - side.abs()) * thick * 0.8, 0.0);
    if along >= 0.0 {
        return tail_knot(creature, knot) + out;
    }
    let forward = tail_knot(creature, 0) - tail_knot(creature, 1);
    let forward = forward / forward.length().max(1e-6);
    tail_knot(creature, 0) + forward * (-along) * shape.tail_step() + out * 1.25
}

/// The train's feathers.
///
/// Five rows rooted one after another from the back down the tail, short coverts nearest the body
/// and the long tail feathers at its end. Each follows the simulated sheet from its own root, as it
/// billows, fanned out across it as its place in the row has it and a little more or less by
/// chance, but always in the sheet's own plane, so the feathers lie over one another as a tail does
/// and never twist through it. Each is narrow at its quill, widest past its middle and rounded at
/// its tip, with barbs angled back from a pale shaft and beads along the shaft lit by the aur sacs.
/// `long` stretches them as an animal's genes have them. Returns the vanes, the shafts and the
/// beads.
pub fn plumage(point: Vec3, creature: &Tetralate, shape: Train, long: f32) -> Vec3 {
    let middle = train_knot(creature, 0.5, 2);
    if (point - middle).length() > shape.scale {
        return Vec3::splat(shape.scale * 0.5);
    }
    let mut vanes = 1e6f32;
    let mut ribs = 1e6f32;
    let mut beads = 1e6f32;
    for feather in 0..TRAIN_FEATHERS {
        let (along, across, reach) = train_feather(feather);
        let reach = reach * (0.92 + creature.chance(620 + feather) * 0.16);
        let base = feather_root(creature, shape, along, across);
        let start = train_knot(creature, across, 0);
        let splay = (across - 0.5) * 0.35 + (creature.chance(600 + feather) - 0.5) * 0.2;
        let last = (((KNOTS - 1) as f32 * reach * long).round() as usize).clamp(2, KNOTS - 1);
        // A feather's width a share of the way along it: a quill at the root, widest beyond the
        // middle, and closing to a rounded tip.
        let width = |knot: usize| {
            let u = knot as f32 / last as f32;
            (0.006 + 0.05 * reach * (u * 3.0).min(1.0) * (1.0 - u.power(3))) * shape.scale
        };
        // Across the sheet at a knot, from its first strand to its last.
        let breadth = |knot: usize| {
            let across = train_knot(creature, 1.0, knot) - train_knot(creature, 0.0, knot);
            across / across.length().max(1e-6)
        };
        let at = |knot: usize| {
            let offset = (train_knot(creature, across, knot) - start) * reach;
            base + offset + breadth(knot) * splay * offset.length()
        };
        let tip = at(last);
        if (point - base.lerp(tip, 0.5)).length()
            > (tip - base).length() * 0.6 + 0.06 * reach * shape.scale
        {
            continue;
        }
        for knot in 0..KNOTS - 1 {
            if knot >= last {
                break;
            }
            let a = at(knot);
            let b = at(knot + 1);
            let normal = (b - a).cross(breadth(knot));
            let normal = normal / normal.length().max(1e-6);
            vanes = vanes.min(feather_strip(
                point,
                a,
                b,
                knot as f32 * shape.train_step() * reach,
                (width(knot), width(knot + 1)),
                normal,
                10.0,
            ));
            ribs = ribs.min(tapered_tail(
                point,
                a,
                b,
                0.005 * shape.scale,
                0.0035 * shape.scale,
            ));
            if reach > 0.8 && knot >= 2 && knot + 1 < last {
                beads = beads.min((point - b).length() - 0.008 * shape.scale);
            }
        }
    }
    vec3(vanes, ribs, beads)
}
