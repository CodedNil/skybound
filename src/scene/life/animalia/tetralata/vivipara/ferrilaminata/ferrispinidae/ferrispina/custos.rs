#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

pub const BIOLOGY: Biology = Biology {
    size: 0.25,
    maturity_days: 2200.0,
    lifespan_days: 42000.0,
    // One to three young, more than its kin, for it guards a nursery rather than chasing herds.
    litter: Litter::new(1, 1, [0.4, 0.4, 0.16, 0.04, 0.0, 0.0]),
    gestation_days: 220.0,
    altitude: (2000.0, 9000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

/// How large the ruff of gill plates at its throat, and how many blades in its rampart: a big
/// ruff warns off more intruders before it has to fight, and fewer, broader blades stand firmer.
pub const RUFF: Gene = Gene(38);
pub const BLADES: Gene = Gene(39);

taxon!(
    Species,
    "Ferrispina custos",
    "Iron Spined Guardians",
    [
        "Iron Spined Guardians are Shredders that turned from hunting to keeping: shorter and deeper chested, with fewer, broader blades standing nearly upright and broad wings for hovering in a stand-off.",
        "They keep nurseries for the young of every kind, not their own alone: sanctuaries where the eggs and young of many species are brought and watched over, which nearly every hunter of the sky has learned to leave alone.",
        "A sanctuary is no trap: any animal may pass through it, hunters among them, but no hunting and no fighting is allowed there. A hunter that starts a hunt within one meets the guardians crest raised, the gold-banded web spread between the blades and the red gill plates flared at the throat, and is shouldered out, unharmed, its hunt broken; banking patrols round the nursery keep most from ever trying.",
        "The trust runs both ways: many animals bring food to the guardians in thanks for the young they keep, so a guardian seldom needs to hunt.",
        "Nurseries most often gather round an ancient Whisperwood, for which the guardians have an affinity no one can explain; the tree itself plays no part. As eggs, young and tree all drift on the wind, the guardians nudge strays gently back to keep them close together.",
        "The display that drives off a hunter, given to one another, decides which guardians pair, and the broadest crests are the most sought after.",
    ],
);

pub fn animate(c: &mut Tetralate) {
    prepare_pose(c, 6.0, 6.5, 0);
    c.opaque_bounds = vec3(6.0, 7.5, 22.0).extend(0.0);
    rig_tentacles(c, mouth(BUILD.length, BUILD.head), 1.1, ORAL_PAIRS);
}

/// A Shredder that stays to fight rather than chase: shorter and much deeper in the chest, so it
/// can hang facing an attacker and turn on the spot, with a shorter tail and broader, rounder wings
/// for hovering. Its crest has fewer, broader blades standing nearly upright, a rampart over the
/// nursery more than a lance, and a tall web between them, gold-banded, that it spreads to warn off
/// anything that comes too near.
const BUILD: Build = Build {
    length: 7.0,
    depth: 1.65,
    head: 0.9,
    blades: 5,
    height: 4.6,
    width: 1.5,
    rake: 0.2,
    web: 0.95,
};

/// The lineage's circlet of eyes round its narrow head and the pair beneath, each seated on its
/// own skin.
#[cfg(not(target_arch = "spirv"))]
pub const ANCHORS: u32 = 5;

/// Where eye `index` lies on its design, before it is seated on its own skin.
#[cfg(not(target_arch = "spirv"))]
pub fn anchor(index: u32, _: &CreatureInstance) -> Vec4 {
    eye(index, BUILD).extend(cowl_eye(index, BUILD.length, BUILD.head).1 * SUNK)
}

pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
    let blades = 4 + (c.gene(BLADES) * 2.99) as u32;
    let build = Build {
        blades,
        width: BUILD.width * 5.0 / blades as f32,
        ..BUILD
    };
    let mut hit = shredder(p, c, build);
    // A ruff of red gill plates at the throat, flared in threat.
    let throat = vec3(p.x.abs(), -p.y, p.z);
    for plate in 0..3 {
        let f = plate as f32;
        hit.blend(
            super::blade(
                throat,
                vec3(0.45, BUILD.depth * 0.35, -BUILD.length * 0.62 + f * 0.7),
                (1.4 - f * 0.25) * c.life.y.max(0.3) * (0.75 + c.gene(RUFF) * 0.5),
                0.35,
                0.7,
                0.7 + f * 0.15,
                // Soft plates, blunt and never worn.
                Vec2::ZERO,
            ),
            2,
            0.08,
        );
    }
    hit.blend(
        oral_tentacles(p, c, mouth(BUILD.length, BUILD.head), 1.1, ORAL_PAIRS),
        8,
        0.1,
    );
    hit
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.26, 0.36, 0.35, 0.0),
    vec3(0.88, 0.62, 0.20),
    [vec4(11.0, 3.4, -1.6, 0.8), vec4(8.0, 2.6, 2.4, 0.8)],
)
.hinged([vec2(1.2, 0.4), vec2(1.1, 0.35)])
.stalked([vec2(0.35, 0.40), vec2(0.35, 0.40)])
.styled([vec4(0.6, 0.7, 4.0, 0.0), vec4(0.5, 0.8, 3.0, 0.0)])
.skinned(vec4(2.0, 0.9, 1.0, 0.10))
.flying(vec4(0.70, 0.36, 0.40, 1.00))
.marked(vec4(2.00, 5.00, 0.80, 0.40), vec3(0.85, 0.60, 0.15));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Iron Spined Guardians keep sanctuaries for the young of every kind. They gather where they
    //! feel at home, most often about an ancient Whisperwood, and patrol the nursery in banking
    //! circuits; the eggs and young of many kinds are kept there, each stray nudged gently back,
    //! and nearly every hunter of the sky knows to hunt elsewhere. Any animal may pass through, but
    //! no hunting is allowed: one that starts a hunt there is met crest first and shouldered out,
    //! unharmed. Guardians seldom hunt: the animals whose young they keep bring them food.

    use super::*;
    use crate::scene::life::{
        animalia::{
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges, orbit,
            },
            nature::HUNT,
            physiology::{Light, Physiology, Take, plant},
            tetralata::vivipara::ferrilaminata::ferrispinidae::ferrispina::ETHOLOGY as IRON_SPINE,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        flocks: false,
        ..super::super::PHYSIOLOGY
    };

    /// How far about a guardian the young of every kind are under its guard.
    const SANCTUARY: f32 = 150.0;

    /// Guardians stand between the young and whatever threatens them rather than flee, break up
    /// any hunt within the sanctuary, and keep the nursery together.
    pub const ETHOLOGY: Ethology = Ethology {
        urges: guard,
        fear: |s, mind: &mut Mind| {
            let fear = (IRON_SPINE.fear)(s, mind);
            if s.seen.mark(YOUNG).is_some() {
                fear * 0.3
            } else {
                fear
            }
        },
        wander: patrol,
        notice,
        shelters: SANCTUARY,
        ..IRON_SPINE
    };

    /// A hunter hunting within the sanctuary.
    const INTRUDER: Interest = Interest("intruder");
    /// An ancient Whisperwood, which guardians keep their nurseries about.
    const NURSERY: Interest = Interest("nursery");

    /// A guardian watches over the eggs and young of every kind, not its own alone, marks any
    /// hunter hunting within its sanctuary, and knows an ancient Whisperwood when it sees one.
    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (IRON_SPINE.notice)(glimpse, seen);
        if glimpse.heard {
            return;
        }
        let other = glimpse.other;
        if plant(other) {
            if other == Kind::Susurrus && glimpse.grown > 10.0 {
                seen.note(NURSERY, glimpse.offset);
            }
            return;
        }
        if glimpse.grown < 0.5 {
            seen.note(YOUNG, glimpse.offset);
        }
        let hunting = matches!(glimpse.activity, Activity::Hunt | Activity::Strike);
        if hunting
            && other != glimpse.kind
            && other.takes(Take::Prey)
            && glimpse.offset.length() < SANCTUARY
        {
            seen.note(INTRUDER, glimpse.offset);
        }
    }

    /// The heart of the nursery: an ancient Whisperwood if there is one, or else where the
    /// guardians gather.
    fn nursery(s: &Situation) -> Vec3 {
        s.seen.mark(NURSERY).or(s.seen.kin).unwrap_or(Vec3::ZERO)
    }

    fn guard(s: &Situation, urges: &mut Urges) {
        (IRON_SPINE.urges)(s, urges);
        let devotion = s.temper(DEVOTION);
        if s.seen.threat.is_some() {
            let young = if s.seen.mark(YOUNG).is_some() {
                1.0
            } else {
                0.4
            };
            // The more devoted the guardian, the less it takes to bring it between the young and
            // what threatens them.
            urges.add(
                GUARD,
                s.threat * (0.6 + s.temper(AGGRESSION) * 0.5 + devotion * 0.7) * young,
            );
        }
        if s.seen.mark(INTRUDER).is_some() {
            urges.add(DRIVE_OFF, 0.7 + s.temper(AGGRESSION) * 0.4 + devotion * 0.5);
        }
        if s.seen
            .mark(YOUNG)
            .is_some_and(|ward| (ward - nursery(s)).length() > 25.0)
        {
            urges.add(TEND, 0.25 + devotion * 0.35);
        }
        // Fed by those whose young it keeps, a guardian hunts only when it must.
        urges.scale(HUNT, 0.5);
    }

    /// Standing between a threat and the young.
    const GUARD: Intent = Intent {
        name: "guard",
        act: |_, s| {
            let stand = s.seen.threat.map(|threat| {
                s.seen
                    .mark(YOUNG)
                    .map_or(threat.offset, |young| (young + threat.offset) * 0.5)
            });
            Goal::new(Activity::Guard, stand, 1.0)
        },
    };

    /// Facing down a hunter hunting in the sanctuary, crest raised, and shouldering it out.
    const DRIVE_OFF: Intent = Intent {
        name: "drive off",
        act: |_, s| Goal::new(Activity::Defend, s.seen.mark(INTRUDER), 1.3),
    };

    /// Nudging a stray egg or young one back towards the rest.
    const TEND: Intent = Intent {
        name: "tend",
        act: |_, s| {
            let centre = nursery(s);
            let goal = s
                .seen
                .mark(YOUNG)
                .map(|ward| ward + (centre - ward).normalize_or_zero() * 8.0);
            Goal::new(Activity::Nudge, goal, 0.4).hovering()
        },
    };

    /// Banking circuits round the nursery, each guardian on its own: they warn off hunters
    /// before any comes near.
    fn patrol(_: &mut Mind, s: &Situation) -> Goal {
        let up = s.up();
        let centre = nursery(s);
        let radius = 40.0 + s.wobble().sin() * 12.0;
        Goal::new(
            Activity::Guard,
            Some(centre + orbit(up, centre, radius, s.now * 0.12 + s.wobble())),
            0.5,
        )
    }
}
