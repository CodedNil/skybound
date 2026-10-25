pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::curved_tube,
    annulated,
    biology::Biology,
    capsule, eyes, mouth, smooth_min,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 90.0,
    lifespan_days: 2000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 36.7;

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
/// It looks nothing like the scavengers it shares a class with: a body thin as
/// a stick, built to hang motionless for hours, a heavy banded silk gland slung
/// beneath it, and its one pair of mouth tentacles grown enormously long and
/// held spread wide, resting on the lines of its web to feel for anything
/// caught. Combs at the tail draw out the silk, which trails behind it in pale
/// strands.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(0.32, 0.3, 2.6), 0.32);
    let gland = annulated(
        p,
        vec3(0.0, -0.2, 0.6),
        vec3(0.0, -0.55, 3.4),
        vec3(0.5, 1.2, 0.3),
        11.0,
        0.1,
        0.95,
    );
    let mut hit = Surface::new(smooth_min(body, gland, 0.25), 0);
    let side = vec3(p.x.abs(), p.y, p.z);
    let lip = mouth(2.6, 0.32);
    let sway = (c.animation.z * 0.6).sin() * 0.08;
    hit.join(
        curved_tube(
            side,
            lip + vec3(0.08, 0.0, 0.0),
            lip + vec3(2.2, 0.4 + sway, -1.2),
            lip + vec3(4.6, -0.4 - sway, -0.8),
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
    for strand in 0..3 {
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
    hit.join(eyes(p, 2.6, 0.32), 3);
    hit
}

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
