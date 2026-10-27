pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{curved_tube, gape},
    annulated,
    biology::Biology,
    capsule, eyes,
    genome::Gene,
    mouth, smooth_min,
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
    size: 0.083,
    maturity_days: 90.0,
    lifespan_days: 2000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 36.7;

/// How heavy a silk gland it carries, how wide it spreads its mouth tentacles on the lines, and how
/// many strands it pays out behind.
///
/// A heavy gland spins more web between meals, a wide span feels more of it, and many strands lay
/// it faster.
pub const GLAND: Gene = Gene(38);
pub const SPAN: Gene = Gene(39);
pub const STRANDS: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Sericifex textor",
    "Weavers",
    [
        "Weavers hold themselves between kelp fronds on four wings while a pair of combs at the tail draws silk from a heavy banded gland and the mouth tentacles fasten it.",
        "Loose catching strands hang across feeding approaches, while stronger braided lines keep neighbouring plants from drifting beyond reach.",
        "They read changes in tension to distinguish grazing visitors from moving anchors, then reel captured prey toward the sheltered centre.",
        "Lines are loosened or replaced as currents shift, allowing the structure to breathe with a Jungle rather than resist it as a rigid net.",
        "Nests, collected bright fragments and the work of Næ crafters accumulate along old anchor corridors, turning a feeding apparatus into a persistent aerial neighbourhood.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.5, 3.5, 9.0).extend(0.0);
}

/// A trap-builder that hunts by waiting.
///
/// It looks nothing like the scavengers it shares a class with: a body thin as a stick, built to
/// hang motionless for hours, a heavy banded silk gland slung beneath it, and its one pair of
/// mouth tentacles grown enormously long and held spread wide, resting on the lines of its web
/// to feel for anything caught. Combs at the tail draw out the silk, which trails behind it in
/// pale strands.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(0.32, 0.3, 2.6), 0.32);
    let gland = annulated(
        p,
        vec3(0.0, -0.2, 0.6),
        vec3(0.0, -0.55, 3.4),
        vec3(0.4 + c.gene(GLAND) * 0.2, 1.0 + c.gene(GLAND) * 0.4, 0.3),
        11.0,
        0.1,
        0.95,
    );
    let mut hit = Surface::new(smooth_min(body, gland, 0.25), 0);
    let side = vec3(p.x.abs(), p.y, p.z);
    let lip = mouth(2.6, 0.32);
    let sway = (c.animation.z * 0.6).sin() * 0.08;
    let span = 3.9 + c.gene(SPAN) * 1.4;
    hit.join(
        curved_tube(
            side,
            lip + vec3(0.08, 0.0, 0.0),
            lip + vec3(span * 0.48, 0.4 + sway, -1.2),
            lip + vec3(span, -0.4 - sway, -0.8),
            0.07,
        ),
        8,
    );
    let start = vec3(0.12, -0.5, 3.3);
    let end = vec3(0.5, -0.8, 4.0);
    let mut comb = capsule(side, start, end, 0.05);
    for tooth in 0..5 {
        let at = start.lerp(end, (tooth as f32 + 0.5) / 5.0);
        comb = comb.min(capsule(side, at, at + vec3(0.08, -0.3, 0.06), 0.02));
    }
    hit.join(comb, 7);
    let strands = 2 + (c.gene(STRANDS) * 3.0) as u32;
    for strand in 0..5 {
        if strand >= strands {
            break;
        }
        let from = vec3(0.3 + strand as f32 * 0.12, -0.9, 4.0);
        let drift = (c.animation.z * 0.7 + strand as f32).sin() * 0.3;
        hit.join(
            capsule(
                side,
                from,
                from + vec3(0.6 + drift, -0.9 - strand as f32 * 0.3, 4.2),
                0.012,
            ),
            4,
        );
    }
    // Two hooked fangs folded back under the mouth, which swing down and forward as it feeds to
    // pierce what it has reeled in and hold it while its gut juices soften it.
    let open = gape(c);
    let (swing, hold) = (open * 1.1).sin_cos();
    let root = lip + vec3(0.07, -0.04, 0.0);
    let fold = |v: Vec3| vec3(v.x, v.y * hold - v.z * swing, v.z * hold + v.y * swing);
    hit.join(
        curved_tube(
            side,
            root,
            root + fold(vec3(0.02, -0.06, 0.14)),
            root + fold(vec3(-0.04, -0.16, 0.22)),
            0.035,
        ),
        7,
    );
    hit.socket(eyes(p, 2.6, 0.32), 0.32 * 0.16);
    hit
}

/// The silk gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const SILK: Organ = Organ::new("silk gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, -0.16, -1.53), vec3(0.08, 0.08, 0.08)),
    site(PUMP, vec3(0.00, -0.16, -0.56), vec3(0.09, 0.09, 0.09)),
    site(CROP, vec3(0.00, -0.34, 0.72), vec3(0.17, 0.19, 0.25)),
    pair(BREATHER, vec3(0.10, -0.19, -0.80), vec3(0.06, 0.06, 0.06)),
    site(FLOAT, vec3(0.00, 0.03, 0.72), vec3(0.23, 0.23, 0.71)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.10, -0.06, -0.16),
        vec3(0.09, 0.10, 0.14),
    ),
    site(CISTERN, vec3(0.00, -0.34, 1.85), vec3(0.20, 0.20, 0.57)),
    site(BROOD, vec3(0.00, -0.16, 2.49), vec3(0.09, 0.10, 0.29)),
    site(SILK, vec3(0.00, -0.84, 2.49), vec3(0.17, 0.17, 0.17)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.35, 0.24, 0.44, 0.0),
    vec3(0.83, 0.58, 0.30),
    [vec4(4.6, 0.9, -0.5, 0.27), vec4(2.2, 0.6, 0.6, 0.27)],
)
.hinged([vec2(0.32, 0.1), vec2(0.3, 0.1)])
.styled([vec4(0.1, 0.8, 0.0, 0.0), vec4(0.1, 0.8, 0.0, 0.0)])
.skinned(vec4(1.0, 0.8, 1.8, 0.00))
.flying(vec4(3.00, 0.12, 0.10, 0.50))
.marked(vec4(2.00, 4.00, 0.90, 0.40), vec3(0.95, 0.85, 0.30));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
