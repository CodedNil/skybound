use super::{Build, shredder};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::Biology,
    mouth, prepare_pose,
    surface::{Ray, Surface, TraceResult},
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
    organs::{Viscera, pair, site},
};
#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    size: 0.25,
    maturity_days: 2200.0,
    lifespan_days: 42000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Ferrispina custos",
    "Iron Spined Guardians",
    [
        "Iron Spined Guardians are Shredders that stay near communal nursery currents: shorter and deeper chested, with fewer, broader blades standing nearly upright and broad wings for hovering in a stand-off.",
        "Faced with a predator they raise the crest, spread the gold-banded web between the blades and flare the red gill plates at the throat, and repeated banking patrols discourage anything from approaching the young.",
        "The same display, given to one another, decides which guardians pair, and the broadest crests are the most sought after.",
        "Vocal exchanges and the motion of nearby adults help maintain a loose moving boundary around the nursery.",
        "The nursery travels with a suitable current, so protection depends on recognising companions rather than defending a fixed place.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 6.0, 10.0, 0);
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

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = shredder(p, c, BUILD);
    // A ruff of red gill plates at the throat, flared in threat.
    let throat = vec3(p.x.abs(), -p.y, p.z);
    for plate in 0..3 {
        let f = plate as f32;
        hit.join(
            super::blade(
                throat,
                vec3(0.45, BUILD.depth * 0.35, -BUILD.length * 0.62 + f * 0.7),
                (1.4 - f * 0.25) * c.life.y.max(0.3),
                0.35,
                0.7,
                0.7 + f * 0.15,
            ),
            2,
        );
    }
    hit.join(
        oral_tentacles(p, c, mouth(BUILD.length, BUILD.head), 1.1, ORAL_PAIRS),
        8,
    );
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.07, -4.91), vec3(0.16, 0.27, 0.39)),
    site(PUMP, vec3(0.00, -0.33, -2.55), vec3(0.18, 0.29, 0.39)),
    site(CROP, vec3(0.00, 0.07, 0.98), vec3(0.54, 0.55, 1.73)),
    pair(BREATHER, vec3(0.24, 0.40, -2.75), vec3(0.19, 0.20, 0.20)),
    site(FLOAT, vec3(0.00, 0.60, -0.98), vec3(0.22, 0.35, 0.40)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.71, 0.40, -1.18),
        vec3(0.18, 0.20, 0.20),
    ),
    site(CISTERN, vec3(0.00, 0.07, 2.95), vec3(0.65, 0.65, 1.37)),
    site(BROOD, vec3(0.00, 0.07, 4.52), vec3(0.37, 0.37, 1.18)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.26, 0.36, 0.35, 0.0),
    vec3(0.88, 0.62, 0.20),
    [vec4(11.0, 3.4, -1.6, 0.8), vec4(8.0, 2.6, 2.4, 0.8)],
)
.hinged([vec2(1.2, 0.4), vec2(1.1, 0.35)])
.styled([vec4(0.6, 0.7, 4.0, 0.0), vec4(0.5, 0.8, 3.0, 0.0)])
.skinned(vec4(2.0, 0.9, 1.0, 0.10))
.flying(vec4(0.70, 0.36, 0.40, 1.00))
.marked(vec4(2.00, 5.00, 0.80, 0.40), vec3(0.85, 0.60, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Mind, Situation, Urges},
            physiology::{Light, Physiology},
            tetralata::vivipara::{
                YOUNG, ferrilaminata::ferrispinidae::ferrispina::ETHOLOGY as IRON_SPINE,
            },
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        flocks: false,
        ..super::super::PHYSIOLOGY
    };

    /// Guardians stand by the young of their kind: faced with a threat they put themselves between
    /// it and the nursery rather than flee, and gang together against it.
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
        ..IRON_SPINE
    };

    fn guard(s: &Situation, urges: &mut Urges) {
        (IRON_SPINE.urges)(s, urges);
        if s.seen.threat.is_some() {
            let young = if s.seen.mark(YOUNG).is_some() {
                1.0
            } else {
                0.4
            };
            urges.add(GUARD, s.threat * (0.8 + s.temper.aggression) * young);
        }
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
}
