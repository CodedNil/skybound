use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::curved_tube,
    biology::Biology,
    capsule, eyes,
    genome::Gene,
    mouth,
    surface::{Ray, Surface, TraceResult},
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
    organs::{Function, Organ, Viscera, pair, site},
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.05,
    maturity_days: 3.0,
    lifespan_days: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 19.8;

/// How wide it spreads its plumes, how finely they are barbed, and how bright the beads along them.
///
/// Wide, fine plumes comb more algae from the air, and bright beads draw more of it in, and more of
/// the hunters too.
pub const PLUMES: Gene = Gene(38);
pub const BARBS: Gene = Gene(39);
pub const GLOW: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Scintilla scintillans",
    "Sparkies",
    [
        "Sparkies appear as moving points of intense colour as vast swarms gather algae and suspended seeds, each holding two feathery filter plumes spread before it, every barb tipped with a point of light.",
        "Drifting algae swim towards the light and the plumes, its ancestral mouth tentacles, comb them out of the air, while two pairs of long, narrow, quick wings let each animal adjust to its neighbours and to small changes in the feeding current.",
        "Their light preserves contact through the dim equatorial skies, but also draws Shimmerays and other small-prey hunters.",
        "Fast growth allows dense swarms to appear around a brief food pulse and disperse before the current empties.",
        "Such gatherings trouble larger animals and Næ settlements while supplying an essential link between microscopic plant life and aerial predators, and their eggs are cast loose to drift up on crowns of fine threads.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.2, 1.2, 2.6).extend(0.0);
}

/// A tiny flyer like a firefly: a small head with domed eyes and feathery antennae, a rounded
/// thorax, and a ringed abdomen whose last segments hold the light. A tiny swarming algae-eater
/// that carries its own light. The lamps sit along the top of its back and at the tail, facing up
/// and back: a swarm keeps together by them in the dim equatorial skies, while a hunter rising from
/// below sees only a dark belly against the dark sky. A living starburst. Its body is nearly a
/// sphere, the least surface for its bulk, since so small an animal dries out fast; from it
/// radiates a burst of fine spines that sieve drifting algae and seeds from the air as it tumbles
/// through a swarm, and the short mouth tentacles sweep the catch from their bases into the mouth
/// beneath. Each spine is tipped with a bead of light, so every animal reads as a scatter of bright
/// points, which is how a swarm keeps together in the dim equatorial skies, and what draws the
/// hunters that follow it. Tiny paddle wings make the constant small corrections a crowded swarm
/// demands. The plumes' sweep from the mouth, out and up and forward.
fn plume(side: f32, t: f32, c: &CreatureInstance) -> Vec3 {
    let root = mouth(0.9, 0.16) + vec3(side * 0.05, 0.0, 0.0);
    let sway = (c.animation.z * 1.7 + side).sin() * 0.06;
    let spread = 0.85 + c.gene(PLUMES) * 0.3;
    let control = vec3(side * 0.45 * spread, 0.35 + sway, -1.55);
    let tip = vec3(side * (0.75 * spread + sway), 0.62, -1.9 - spread * 0.18);
    root * (1.0 - t) * (1.0 - t) + control * 2.0 * (1.0 - t) * t + tip * t * t
}

/// A tiny swarm-feeder that fishes with light.
///
/// Its one pair of mouth tentacles has become two feathery filter plumes, held spread before it and
/// tipped along every barb with points of light: the drifting algae it eats swim towards light, and
/// the plumes comb them out of the air as they come. The same lights hold a swarm together through
/// the dim phase, and draw the hunters that follow it. Behind the plumes the body is slim and
/// smooth, and long, narrow wings beat fast enough to hold it steady among its neighbours.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.2, 0.18, 0.9), 0.16), 0);
    let side = if p.x < 0.0 { -1.0 } else { 1.0 };
    let root = plume(side, 0.0, c);
    hit.join(
        curved_tube(
            p,
            root,
            vec3(side * 0.45 * (0.85 + c.gene(PLUMES) * 0.3), 0.35, -1.55),
            plume(side, 1.0, c),
            0.03,
        ),
        4,
    );
    let mut barbs = 1e6f32;
    let mut lights = 1e6f32;
    // Finely barbed plumes carry more, shorter barbs along the same length.
    let count = 5 + (c.gene(BARBS) * 4.99) as u32;
    let bead = 0.022 + c.gene(GLOW) * 0.018;
    for barb in 0..9 {
        if barb >= count {
            break;
        }
        let t = 0.3 + barb as f32 * 0.7 / count as f32;
        let at = plume(side, t, c);
        let alternate = if barb % 2 == 0 { 1.0 } else { -1.0 };
        let fine = 1.15 - c.gene(BARBS) * 0.3;
        let tip = at + vec3(side * 0.18, 0.12 * alternate, -0.08) * (1.2 - t * 0.6) * fine;
        barbs = barbs.min(capsule(p, at, tip, 0.011));
        lights = lights.min((p - tip).length() - bead);
    }
    lights = lights.min((p - plume(side, 1.0, c)).length() - bead * 1.6);
    hit.join(barbs, 4);
    hit.join(lights, 9);
    hit.socket(eyes(p, 0.9, 0.16), 0.16 * 0.16);
    hit
}

/// The spark gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const PLUME: Organ = Organ::new("spark gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.01, -0.72), vec3(0.04, 0.03, 0.06)),
    site(PUMP, vec3(0.00, -0.03, -0.35), vec3(0.05, 0.03, 0.09)),
    site(CROP, vec3(0.00, 0.01, 0.16), vec3(0.09, 0.09, 0.29)),
    pair(BREATHER, vec3(0.17, -0.02, -0.42), vec3(0.02, 0.02, 0.02)),
    site(FLOAT, vec3(0.00, 0.08, -0.16), vec3(0.05, 0.04, 0.05)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.17, 0.02, -0.14),
        vec3(0.05, 0.03, 0.05),
    ),
    site(CISTERN, vec3(0.00, 0.01, 0.49), vec3(0.09, 0.09, 0.20)),
    site(BROOD, vec3(0.00, 0.01, 0.72), vec3(0.05, 0.03, 0.10)),
    site(PLUME, vec3(0.00, -0.01, -0.81), vec3(0.04, 0.03, 0.06)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.22, 0.71, 0.79, 0.25),
    vec3(0.76, 0.98, 0.42),
    [vec4(2.6, 0.45, -0.15, 0.3), vec4(2.2, 0.4, 0.35, 0.3)],
)
.hinged([vec2(0.22, 0.06), vec2(0.2, 0.06)])
.styled([vec4(1.2, 0.15, 0.0, 0.0), vec4(1.2, 0.15, 0.0, 0.0)])
.framed([0.25, 0.25])
.skinned(vec4(5.0, 0.6, 2.0, -0.1))
.flying(vec4(8.00, 0.50, 0.25, 0.60))
.marked(vec4(0.00, 4.00, 0.00, 0.40), vec3(0.80, 0.90, 0.50));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Brood, Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        brood: Brood::Drifting,
        ..super::super::PHYSIOLOGY
    };
}
