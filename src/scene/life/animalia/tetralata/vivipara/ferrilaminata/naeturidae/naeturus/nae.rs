use super::{shaft, vane};
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

/// How tall the soft crest on its crown, how large the aur sacs on its flanks, and how plump its
/// body.
///
/// A family's crests run alike, so kin know one another at a glance, while large sacs heal it
/// faster and a plump body carries it further between meals.
pub const CREST: Gene = Gene(38);
pub const SACS: Gene = Gene(39);
pub const HEFT: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

/// Four pairs of mouth tentacles, the outer pair grown long for tools: the Næ's hands.
const ORAL_PAIRS: u32 = 4;

taxon!(
    Species,
    "Nae",
    "Næ",
    [
        "Næ are small, intelligent four-winged animals with large heads, round bodies and long gripping tails that curl at the tip.",
        "Two large eyes look forwards and down from under a pale brow, the downward pair beneath the chin watching the aur below, and the face they make is the first thing a Nætu learns to know.",
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs, with long outer tips holding tools while shorter tips steady objects and guide food.",
        "Their wings are small and rounded on narrow stalks that twist through a quick stroke, for hovering close to work rather than for distance; a Næ travels far on its Nætu.",
        "Moisture-recovery folds support short flights, although active travel still requires drinks from living basins.",
        "Isolated aur sacs support their magical abilities and are replenished through the protected life they share with bonded Nætu.",
        "They take iron in their diet as all their order does, and their bodies sheathe their aur sacs in it as their Nætu do; what they gather beyond that they work by hand into clasps and tools.",
        "Families groom their companions, tend sheltered homes among dorsal fibres and navigate between drifting communities by currents and the distant polar beams.",
    ],
);

/// The head: its centre and half extents. A Næ is mostly head, as the young of every tetralate
/// are, and keeps that build all its life.
const HEAD: Vec3 = vec3(0.0, 0.03, -0.16);
const HEAD_SIZE: Vec3 = vec3(0.165, 0.15, 0.155);

/// The body behind it, plump round the crop.
const BODY: Vec3 = vec3(0.0, -0.005, 0.08);
const BODY_SIZE: Vec3 = vec3(0.15, 0.14, 0.2);

/// Where the mouth opens, beneath the muzzle, and its tentacles with it.
const MOUTH: Vec3 = vec3(0.0, -0.1, -0.27);

/// How far the mouth tentacles reach, as a share of the lineage's usual reach.
const REACH: f32 = 0.15;

/// A point on the head's surface, `inset` of the way out along `direction` from its centre.
fn on_head(direction: Vec3, inset: f32) -> Vec3 {
    HEAD + HEAD_SIZE * (direction / direction.length()) * inset
}

pub fn animate(creature: &mut Tetralate) {
    prepare_pose(creature, 0.4, 0.8, 0);
    creature.opaque_bounds = vec3(0.5, 0.5, 1.2).extend(0.0);
    rig_tentacles(creature, MOUTH, REACH, ORAL_PAIRS);
}

/// Where its tail runs: back from the body and down a little, swaying as it flies, and curling
/// up at the tip into the hook it grips with.
fn tail_joint(creature: &Tetralate, index: u32) -> Vec3 {
    let mut point = vec3(0.0, -0.01, 0.24);
    let mut angle = -0.15f32;
    for step in 0..index {
        let t = (step + 1) as f32 / 7.0;
        angle += 0.04 + (t - 0.6).max(0.0) * 1.1;
        let sway = (creature.animation.z * 2.2 - t * 4.0).sin() * t * 0.05;
        point += vec3(sway, angle.sin(), angle.cos()) * 0.12;
    }
    point
}

/// A Næ: small, round and mostly head.
///
/// A big soft head under a pale brow, its two large eyes set low to look forwards and down, with
/// the small muzzle beneath them where the mouth tentacles hang, short and curling: its hands. A
/// plump body behind carries the aur sacs glowing on its flanks and the breathing louvres just
/// behind the head, and a soft crest rises from its crown. Its long tail curls up at the tip into
/// a gripping hook, flighted near the middle with a pair of feather vanes. It wears one thing it
/// made: a woven girth strap with a bronze clasp on the back.
pub fn surface(point: Vec3, creature: &Tetralate) -> Surface {
    let head = rounded(point - HEAD, HEAD_SIZE);
    let heft = 0.94 + creature.gene(HEFT) * 0.12;
    let body = rounded(point - BODY, BODY_SIZE * vec3(heft, heft, 1.0));
    let belly = rounded(point - vec3(0.0, -0.05, 0.07), vec3(0.13, 0.1, 0.15) * heft);
    let mut d = smooth_min(smooth_min(head, body, 0.07), belly, 0.05);
    // The brow: a soft pale hood over the eyes, shading them from the glow below.
    let brow = rounded(point - vec3(0.0, 0.1, -0.2), vec3(0.15, 0.05, 0.12));
    d = smooth_min(d, brow, 0.05);
    let muzzle = rounded(point - vec3(0.0, -0.055, -0.285), vec3(0.068, 0.048, 0.045));
    d = smooth_min(d, muzzle, 0.04);
    let crest_height = creature.gene(CREST);
    let crest = rounded(
        point - vec3(0.0, 0.17 + crest_height * 0.02, -0.11 + point.y * 0.4),
        vec3(0.012, 0.045 + crest_height * 0.035, 0.11),
    );
    d = smooth_min(d, crest, 0.03);
    d = louvres(d, point - vec3(0.0, 0.0, 0.06), vec3(0.16, 0.14, 0.2));
    let tail = shaft(point, |index| tail_joint(creature, index), 8, 0.05, 0.014);
    d = smooth_min(d, tail, 0.05);
    let mut hit = with_wings(Surface::new(d, 0), point, creature, LOOK);
    // A pair of feather vanes fanned from the middle of the tail, to steer by.
    let base = tail_joint(creature, 3);
    let onward = tail_joint(creature, 4) - base;
    let onward = onward / onward.length().max(1e-4);
    for side in 0..2 {
        let sign = if side == 0 { -1.0 } else { 1.0 };
        let direction = vec3(sign * 0.7, onward.y * 0.5, 0.7);
        hit.join(
            vane(point, base, direction / direction.length(), 0.24, 0.045),
            1,
        );
    }
    // Two large eyes, set low in the face under the brow to look forwards and down, and the small
    // downward pair beneath the chin that watches the aur.
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let eye = (mirrored - on_head(vec3(0.42, -0.1, -0.9), 0.9)).length() - 0.042;
    hit.socket(eye, 0.042);
    let low = (mirrored - on_head(vec3(0.35, -0.85, -0.42), 0.97)).length() - 0.014;
    hit.socket(low, 0.014);
    // An aur sac on each flank, glowing through the iron collar its body laid down round it.
    let sac = 0.03 + creature.gene(SACS) * 0.02;
    let at = mirrored - vec3(0.14, 0.0, 0.08);
    hit.join(at.length() - sac, 9);
    let sheath = sac * (1.08 + creature.stores.x.min(1.0) * 0.2);
    let collar = (at.length() - sheath)
        .max(-(at.length() - sac * 0.98))
        .max(at.x - sac * 0.3);
    hit.join(collar, 11);
    // The woven strap round its girth, lying on the skin, and its bronze clasp on the back.
    let strap = (body - 0.012).max((point.z - 0.17).abs() - 0.018);
    hit.join(strap, 5);
    hit.join(
        rounded(point - vec3(0.0, 0.138, 0.17), vec3(0.028, 0.012, 0.024)),
        7,
    );
    hit.blend(
        oral_tentacles(point, creature, MOUTH, REACH, ORAL_PAIRS),
        2,
        0.03,
    );
    hit
}

pub fn render_opaque(ray: Ray, creature: &Tetralate) -> Hit {
    trace_material(ray, creature, surface, material)
}

fn material(point: Vec3, id: u32) -> Material {
    // A pale back and brow over a deep teal belly, the countershading of a world lit from below,
    // and a darker mask round the eyes that makes a face of them.
    let back = point.y.smoothstep(-0.02, 0.09);
    let belly = (-point.y).smoothstep(0.02, 0.12);
    let mask = (1.0
        - (vec2(point.x.abs() - 0.065, point.y - 0.0).length() / 0.075).clamp(0.0, 1.0))
        * (-point.z).smoothstep(0.18, 0.26);
    let mut m = Material::skin(
        vec3(0.22, 0.50, 0.52)
            .lerp(vec3(0.88, 0.82, 0.64), back)
            .lerp(vec3(0.05, 0.15, 0.19), belly * 0.85)
            .lerp(vec3(0.10, 0.26, 0.30), mask * 0.7),
        0.55,
    );
    m.texture = WRINKLED;
    m.relief = 0.18;
    m.grain = 1.6;
    m.scatter = 0.6;
    match id {
        2 => {
            // The mouth tentacles, rosy at the tips.
            m.color = vec3(0.30, 0.52, 0.52).lerp(
                vec3(0.92, 0.62, 0.52),
                (-point.y - 0.12).smoothstep(0.0, 0.3),
            );
            m.roughness = 0.4;
            m.translucency = 0.3;
        }
        1 => {
            // Feather vanes, teal banded with gold.
            m.color = vec3(0.12, 0.38, 0.40).lerp(
                vec3(0.86, 0.66, 0.30),
                (point.z * 40.0).sin().smoothstep(0.2, 0.8),
            );
            m.roughness = 0.7;
        }
        3 => {
            // A glossy lens over an amber iris round a wide dark pupil.
            let eye = vec3(point.x.abs(), point.y, point.z) - on_head(vec3(0.42, -0.1, -0.9), 0.9);
            let iris = (eye.length() / 0.042 - 0.0).clamp(0.0, 1.0);
            let pupil = iris.smoothstep(0.55, 0.62);
            m.color = vec3(0.01, 0.015, 0.02).lerp(
                vec3(0.62, 0.42, 0.12),
                pupil * (1.0 - iris.smoothstep(0.85, 0.95)),
            );
            m.roughness = 0.04;
            m.relief = 0.0;
            m.scatter = 0.0;
        }
        5 => {
            let stitch = (point.x.abs() - 0.1).abs().smoothstep(0.004, 0.012);
            m.color = vec3(0.74, 0.49, 0.18).lerp(vec3(0.035, 0.10, 0.16), stitch);
            m.roughness = 0.85;
            m.scatter = 0.0;
        }
        7 => {
            m.color = vec3(0.84, 0.57, 0.18);
            m.metal = 0.65;
            m.roughness = 0.25;
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
    vec3(0.115, 0.0, 0.08),
    vec3(0.025, 0.025, 0.025),
)];

/// Its four small rounded wings, translucent teal with a gold eyespot near each tip, on narrow
/// stalks that twist through a quick, shallow stroke: a hoverer's wings.
pub const LOOK: Look = Look::coat(
    vec4(0.22, 0.50, 0.52, 0.0),
    vec3(0.34, 0.66, 0.68),
    [vec4(0.72, 0.3, -0.02, 0.55), vec4(0.52, 0.24, 0.15, 0.55)],
)
.hinged([vec2(0.12, 0.07), vec2(0.12, 0.04)])
.stalked([vec2(0.22, 0.65), vec2(0.22, 0.65)])
.styled([vec4(0.35, 0.4, 0.0, 0.0), vec4(0.25, 0.4, 0.0, 0.0)])
.framed([0.35, 0.3])
.flying(vec4(3.2, 0.5, 0.35, 1.1))
.marked(vec4(2.0, 1.0, 0.55, 0.45), vec3(0.95, 0.78, 0.40));

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
