use super::{TailFeathers, render_tail_feathers, tail_shaft, wings};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth, prepare_pose, rounded, smooth_min,
    surface::{Hit, Material, Ray, Surface, TraceResult, WRINKLED, trace_material},
    torso,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::tetralata::CISTERN,
    animalia::tetralata::FLIGHT_MUSCLE,
    animalia::tetralata::vivipara::ferrilaminata::naeturidae::AUR_SAC,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 5500.0,
    lifespan_days: 44000.0,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 3.2;

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

/// Four pairs of mouth tentacles, the outer pair grown long for tools: the Næ's hands.
const ORAL_PAIRS: u32 = 4;

taxon!(
    Species,
    "Nae",
    "Næ",
    [
        "Næ are small, intelligent four-winged animals with pear-shaped bodies, swept crests and long gripping tails.",
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs, with long outer tips holding tools while shorter tips steady objects and guide food.",
        "Moisture-recovery folds support short flights, although active travel still requires drinks from living basins.",
        "Isolated aur sacs support their magical abilities and are replenished through the protected life they share with bonded Nætu.",
        "Families groom their companions, tend sheltered homes among dorsal fibres and navigate between drifting communities by currents and the distant polar beams.",
    ],
);

/// The Næ's body: its half-length and the size of its cowl.
const LENGTH: f32 = 0.36;

const HEAD: f32 = 0.14;

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 0.63, 1.5, 0);
    creature.opaque_bounds = vec3(0.9, 1.2, 2.4).extend(0.0);
    rig_tentacles(creature, mouth(LENGTH, HEAD), 0.5, ORAL_PAIRS);
}

/// A Næ: a small tetralate laid along its flight like all its kin.
///
/// Its body is pear-shaped, heavy behind where the crop sits and narrowing to the cowl, with eyes
/// round the rim and a downward pair beneath. Under the cowl hang the four pairs of mouth
/// tentacles that are its hands, the long outer pair for tools and the short inner ones to steady
/// things and feed. A swept crest runs back from the cowl, aur sacs glow on its flanks, and its
/// long gripping tail trails behind. It wears one thing it made: a woven girth strap with a bronze
/// clasp on the back.
pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let body = torso(point, vec3(0.2, 0.17, LENGTH), HEAD);
    let crop = rounded(point - vec3(0.0, -0.03, 0.14), vec3(0.27, 0.22, 0.25));
    let crest = rounded(point - vec3(0.0, 0.17, -0.2), vec3(0.016, 0.07, 0.24));
    let mut hit = Surface::new(smooth_min(smooth_min(body, crop, 0.1), crest, 0.03), 0);
    hit.socket(eyes(point, LENGTH, HEAD), HEAD * 0.16);
    let side = vec3(point.x.abs(), point.y, point.z);
    hit.join((side - vec3(0.25, 0.04, 0.06)).length() - 0.045, 9);
    let strap = vec2(
        vec2(point.x * 0.95, point.y + 0.02).length() - 0.255,
        point.z - 0.0,
    )
    .length()
        - 0.016;
    hit.join(strap, 5);
    hit.join(
        rounded(point - vec3(0.0, 0.215, 0.0), vec3(0.035, 0.012, 0.03)),
        7,
    );
    hit.join(
        oral_tentacles(point, creature, mouth(LENGTH, HEAD), 0.5, ORAL_PAIRS),
        2,
    );
    let tail = tail_shaft(point, creature, vec3(0.20, 0.25, 1.0), 0.07);
    hit.join(tail.distance, tail.material);
    hit
}

pub fn render_opaque(ray: Ray, creature: &CreatureInstance) -> Hit {
    let body = trace_material(ray, creature, surface, material);
    let feathers = render_tail_feathers(
        Ray {
            max_distance: body.distance().min(ray.max_distance),
            ..ray
        },
        creature,
        TailFeathers {
            scale: vec3(0.20, 0.25, 1.0),
            length: 0.38,
            width: 0.085,
        },
        material,
    );
    if feathers.is_hit() { feathers } else { body }
}

fn material(point: Vec3, id: u32) -> Material {
    // A pale cowl over a darker belly: the lineage's countershading, turned over for a world lit
    // from below.
    let mask = (-point.z).smoothstep(0.36, 0.5) * point.y.smoothstep(-0.03, 0.06);
    let belly = (-point.y).smoothstep(0.0, 0.2);
    let mut m = Material::skin(
        vec3(0.16, 0.38, 0.40)
            .lerp(vec3(0.80, 0.74, 0.52), mask)
            .lerp(vec3(0.05, 0.13, 0.16), belly * 0.6),
        0.56,
    );
    m.texture = WRINKLED;
    m.relief = 0.3;
    m.grain = 1.6;
    match id {
        2 => {
            m.color = vec3(0.34, 0.58, 0.54)
                .lerp(vec3(0.84, 0.65, 0.40), (-point.z).smoothstep(0.25, 0.7));
            m.roughness = 0.4;
            m.translucency = 0.25;
        }
        1 => {
            m.color = vec3(0.10, 0.35, 0.37).lerp(
                vec3(0.76, 0.54, 0.24),
                (point.z * 24.0).sin().smoothstep(0.0, 0.7),
            );
            m.roughness = 0.7;
        }
        3 => {
            m.color = vec3(0.006, 0.014, 0.021);
            m.roughness = 0.075;
        }
        4 => {
            m.color = vec3(0.23, 0.16, 0.12);
        }
        5 => {
            let stitch = ((point.x.abs() - 0.17).abs()).smoothstep(0.006, 0.018);
            m.color = vec3(0.74, 0.49, 0.18).lerp(vec3(0.035, 0.10, 0.16), stitch);
            m.roughness = 0.85;
        }
        7 => {
            m.color = vec3(0.84, 0.57, 0.18);
            m.metal = 0.65;
            m.roughness = 0.25;
        }
        8 => {
            m.color = vec3(0.17, 0.075, 0.055);
            m.roughness = 0.3;
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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.03, -0.41), vec3(0.04, 0.04, 0.04)),
    site(PUMP, vec3(0.00, -0.05, -0.06), vec3(0.06, 0.05, 0.08)),
    site(CROP, vec3(0.00, -0.01, 0.32), vec3(0.06, 0.06, 0.06)),
    pair(BREATHER, vec3(0.14, -0.02, -0.04), vec3(0.06, 0.05, 0.06)),
    site(FLOAT, vec3(0.00, 0.10, 0.28), vec3(0.04, 0.04, 0.04)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.11, 0.06, 0.13),
        vec3(0.06, 0.05, 0.07),
    ),
    site(CISTERN, vec3(0.00, -0.03, 0.58), vec3(0.03, 0.03, 0.03)),
    site(BROOD, vec3(0.00, -0.01, 0.49), vec3(0.04, 0.04, 0.04)),
    pair(AUR_SAC, vec3(0.14, -0.02, 0.30), vec3(0.03, 0.03, 0.03)),
]);

/// Drawn by its own tracers.
pub const LOOK: Look = Look::OWN;

pub fn render(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::opaque(render_opaque(ray, creature));
    let thin = render_thin(ray, creature);
    for i in 0..thin.count as usize {
        result.add(thin.layers[i]);
    }
    result
}

pub fn render_thin(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::MISS;
    visit_thin(ray, creature, |hit| result.add(hit));
    result
}

pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, creature: &CreatureInstance, mut emit: F) {
    wings(
        ray,
        creature,
        vec3(0.18, 0.05, -0.08),
        Vec2::new(1.15, 0.42),
        vec3(0.24, 0.53, 0.48),
        &mut emit,
    );
    wings(
        ray,
        creature,
        vec3(0.17, 0.03, 0.15),
        Vec2::new(0.80, 0.30),
        vec3(0.31, 0.34, 0.54),
        &mut emit,
    );
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
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                display_near,
            },
            physiology::Physiology,
            tetralata::{
                FOLLOW,
                vivipara::{CALL, notice as hear},
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
                (0.2 + s.temper.sociability * 0.4) * content * water,
            );
        }
    }

    /// Parted from its family, a Næ asks its Nætu to call for it, and makes for the answer.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        let goal = (super::super::ETHOLOGY.act)(mind, s);
        let apart = s.seen.kin.is_none_or(|kin| kin.length() > 300.0);
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
