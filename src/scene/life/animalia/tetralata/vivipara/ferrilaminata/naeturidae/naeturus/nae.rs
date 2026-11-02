use super::{feather, plume, shaft};
use crate::scene::life::{
    Look, Tetralate,
    animalia::{
        tetralata::{oral_tentacles, rig_tentacles},
        visit_wings, with_wings,
    },
    biology::{Biology, Litter},
    genome::Gene,
    louvres, prepare_pose, rounded, smooth_min,
    surface::{Hit, Material, Ray, Surface, TraceResult, WRINKLED, trace_material},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::tetralata::vivipara::ferrilaminata::naeturidae::AUR_SAC,
    organs::{Part, pair},
};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
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

/// How tall the fin along its back, how large the aur sacs on its flanks, how plump its body, how
/// long its train, and how full the plumes behind its cheeks.
///
/// A family's fins run alike, so kin know one another at a glance; large sacs heal it faster and a
/// plump body carries it further between meals; a long train of bright eyes wins a mate; and full
/// plumes breathe more of the air, for longer flights between drinks.
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
        "Næ are small, intelligent four-winged animals with large heads, soft round bodies, a low fin along the back and a long tail that droops under a train of eyed feathers.",
        "Two large eyes look forwards and down from under a pale brow, the downward pair beneath the chin watching the aur below, and the face they make is the first thing a Nætu learns to know.",
        "Behind each cheek three feathery plumes fan back, the exchange frills of their genus, which take in the air they fly through; a Næ flushes them when it is glad.",
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs, with long outer tips holding tools while shorter tips steady objects and guide food.",
        "Their wings are small, rounded and feathered, on narrow stalks that twist through a quick stroke, for hovering close to work rather than for distance; a Næ travels far on its Nætu.",
        "The bare tip of the tail curls into a hook below the train, and with it a Næ holds fast to the fibres of its Nætu's back through any turn; the train itself is for one another, spread and shivered in greeting and courtship.",
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

/// Where the mouth opens, just inside the chin, and its tentacles with it.
const MOUTH: Vec3 = vec3(0.0, -0.06, -0.24);

/// How far the mouth tentacles reach, as a share of the lineage's usual reach.
const REACH: f32 = 0.15;

/// A point on the head's surface, `inset` of the way out along `direction` from its centre.
fn on_head(direction: Vec3, inset: f32) -> Vec3 {
    HEAD + HEAD_SIZE * (direction / direction.length()) * inset
}

pub fn animate(creature: &mut Tetralate) {
    prepare_pose(creature, 0.4, 0.8, 0);
    creature.opaque_bounds = vec3(0.7, 0.9, 1.4).extend(0.0);
    rig_tentacles(creature, MOUTH, REACH, ORAL_PAIRS);
}

/// Where its tail runs: back from the body and drooping, swaying as it flies, and curling forward
/// at the tip into the hook it holds on with.
fn tail_joint(creature: &Tetralate, index: u32) -> Vec3 {
    let mut point = vec3(0.0, -0.02, 0.2);
    let mut angle = 0.12f32;
    for step in 0..index {
        let t = (step + 1) as f32 / 7.0;
        angle += 0.05 + (t - 0.8).max(0.0) * 6.0;
        let sway = (creature.animation.z * 1.8 - t * 3.0).sin() * t * 0.04;
        point += vec3(sway, -angle.sin(), angle.cos()) * 0.11;
    }
    point
}

/// A Næ: small, soft and mostly head.
///
/// Its big head runs without a waist into a plump body, under a low fin that rises from the crown
/// and runs down the back. Two large eyes are set low under a pale brow to look forwards and down,
/// and below them the mouth tentacles hang from the chin, short and curling: its hands. Three
/// feathery plumes fan back behind each cheek, threaded with rings of worked iron, and the aur
/// sacs glow on its flanks. The tail droops under a train of long eyed feathers, its bare tip
/// curled into a hook beneath them.
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
    // A soft pale brow over the eyes, shading them from the glow below.
    let brow = rounded(mirrored - vec3(0.06, 0.09, -0.22), vec3(0.08, 0.045, 0.1));
    d = smooth_min(d, brow, 0.05);
    let muzzle = rounded(point - vec3(0.0, -0.05, -0.28), vec3(0.075, 0.05, 0.05));
    d = smooth_min(d, muzzle, 0.05);
    // Full cheeks below and behind the eyes, where the plumes are rooted.
    let cheek = rounded(mirrored - vec3(0.12, -0.03, -0.16), vec3(0.07, 0.07, 0.08));
    d = smooth_min(d, cheek, 0.05);
    // A low fin from the crown down the back, as tall as the family runs.
    let height = 0.03 + creature.gene(CREST) * 0.03;
    let fin = rounded(
        point
            - vec3(
                0.0,
                0.13 + height * 0.5 - (point.z + 0.05).max(0.0) * 0.12,
                -0.02,
            ),
        vec3(0.008, height, 0.2),
    );
    d = smooth_min(d, fin, 0.025);
    d = louvres(d, point - vec3(0.0, 0.0, 0.08), vec3(0.15, 0.13, 0.18));
    let tail = shaft(point, |index| tail_joint(creature, index), 8, 0.06, 0.016);
    d = smooth_min(d, tail, 0.06);
    let mut hit = with_wings(
        Surface::new(d, if fin < d + 0.004 { 15 } else { 0 }),
        point,
        creature,
        LOOK,
    );
    // Two large eyes, set low in the face under the brow to look forwards and down, and the small
    // downward pair beneath the chin that watches the aur.
    let eye = (mirrored - on_head(vec3(0.42, -0.1, -0.9), 0.95)).length() - 0.042;
    hit.socket(eye, 0.042);
    let low = (mirrored - on_head(vec3(0.35, -0.85, -0.42), 0.97)).length() - 0.014;
    hit.socket(low, 0.014);
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
    hit.blend(
        oral_tentacles(point, creature, MOUTH, REACH, ORAL_PAIRS),
        2,
        0.05,
    );
    hit
}

/// The three feathery plumes behind each cheek, sweeping back and up, with a ring of worked iron
/// on each stem.
fn plumes(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    if (mirrored - vec3(0.25, 0.02, -0.04)).length() > 0.5 {
        return hit;
    }
    let full = 0.85 + creature.gene(PLUMES) * 0.3;
    // They stir with the wingbeat.
    let stir = (creature.animation.z * 3.0).sin() * 0.015;
    for index in 0..3 {
        let lane = index as f32;
        let root = vec3(0.15, 0.05 - lane * 0.05, -0.13 + lane * 0.02);
        let bend = root + vec3(0.1, 0.08 - lane * 0.06 + stir, 0.05) * full;
        let tip = bend + vec3(0.07, 0.07 - lane * 0.07 + stir, 0.16 - lane * 0.03) * full;
        hit.join(
            plume(mirrored, root, bend, tip, 0.014, 0.05 * full, 8.0),
            16,
        );
        // A ring of worked iron round the stem, given by a friend.
        let at = root.lerp(bend, 0.55);
        let axis = (bend - root) / (bend - root).length();
        let q = mirrored - at;
        let along = q.dot(axis);
        let around = (q - axis * along).length();
        let ring = vec2(around - 0.012, along).length() - 0.0045;
        hit.join(ring, 7);
    }
    hit
}

/// The train: long eyed feathers fanned from the tail and drooping under it, longest towards its
/// end, over the bare hooked tip.
fn train(point: Vec3, creature: &Tetralate, mut hit: Surface) -> Surface {
    let middle = tail_joint(creature, 4);
    if (point - middle).length() > 1.1 {
        return hit;
    }
    let long = 0.85 + creature.gene(TRAIN) * 0.3;
    let mut vanes = 1e6f32;
    let mut rings = 1e6f32;
    let mut centres = 1e6f32;
    for joint in 1..5 {
        let root = tail_joint(creature, joint);
        let ahead = tail_joint(creature, joint + 1) - root;
        let ahead = ahead / ahead.length();
        let reach = (0.4 + joint as f32 * 0.1) * long;
        for lane in 0..4 {
            let spread = (lane as f32 - 1.5) * (0.28 + joint as f32 * 0.07);
            // Drooping a little more the further out they reach, as a heavy train does.
            let direction = ahead + vec3(spread, -0.12 - spread.abs() * 0.15, 0.0);
            let parts = feather(
                point,
                root,
                direction / direction.length(),
                reach,
                0.075 + joint as f32 * 0.01,
            );
            vanes = vanes.min(parts.x);
            rings = rings.min(parts.y);
            centres = centres.min(parts.z);
        }
    }
    hit.join(vanes, 1);
    hit.join(rings, 13);
    hit.join(centres, 14);
    // The tail's quills, where the feathers root, a little thicker than the shaft.
    let quill = tapered_tail(
        point,
        tail_joint(creature, 1),
        tail_joint(creature, 5),
        0.03,
        0.022,
    );
    hit.blend(quill, 0, 0.03);
    hit
}

pub fn render_opaque(ray: Ray, creature: &Tetralate) -> Hit {
    trace_material(ray, creature, surface, material)
}

fn material(point: Vec3, id: u32) -> Material {
    // A pale back and brow over a deep teal belly, the countershading of a world lit from below.
    let back = point.y.smoothstep(-0.02, 0.09);
    let belly = (-point.y).smoothstep(0.02, 0.12);
    // The muzzle and chin a warm cream, which makes a face of the eyes above them.
    let muzzle = (-point.z).smoothstep(0.22, 0.3) * (-point.y).smoothstep(-0.02, 0.04);
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
            // A feather's eye: a ring of gold.
            m.color = vec3(0.92, 0.70, 0.26);
            m.roughness = 0.35;
            m.film = 0.8;
            m.relief = 0.0;
        }
        14 => {
            // The eye's centre: deep blue, shot with violet as the light moves over it.
            m.color = vec3(0.10, 0.16, 0.48);
            m.roughness = 0.25;
            m.film = 1.0;
            m.relief = 0.0;
        }
        15 => {
            // The fin, pale and thin enough to glow at its edge.
            m.color = vec3(0.62, 0.84, 0.80);
            m.translucency = 0.5;
            m.roughness = 0.4;
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
            // A glossy lens over an amber iris round a wide dark pupil, measured across the eye's
            // face from the line it looks along.
            let looking = vec3(0.42, -0.1, -0.9).normalize();
            let eye = vec3(point.x.abs(), point.y, point.z) - on_head(looking, 0.95);
            let across = (eye - looking * eye.dot(looking)).length() / 0.042;
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
