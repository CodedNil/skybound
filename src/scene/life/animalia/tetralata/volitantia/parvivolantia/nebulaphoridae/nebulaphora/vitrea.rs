use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    eyes, mouth, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail, torso,
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 45.0,
    lifespan_days: 365.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 19.8;

taxon!(
    Species,
    "Nebulaphora vitrea",
    "Mistings",
    [
        "Mistings spread four broad, transparent, coloured wings, the front pair swept forward and the rear pair back, into a near disc around a slight central body, catching scattered aur light along fine supporting veins.",
        "Each wing pair adjusts separately as they hover beside a swaying Bloom.",
        "In place of the ancestral tentacle pair a single long feeding tube, kept coiled like a spring beneath the cowl, unrolls to reach the nectar without the animal landing on the petals.",
        "Pollen clings to a fuzzy collar behind the head and travels with the next visit, linking flowers across open gaps in the Gardens.",
        "They favour sheltered flowers along humid Swampland margins, where rainfall provides moisture without requiring an approach to the lethal geyser jets.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(2.0, 2.1, 5.4).extend(0.0);
}

/// A hovering flower.
///
/// Its four wings are huge, round and translucent, the front pair swept forward
/// and the rear pair back, so together they open into a near disc that reads as
/// a bloom in its own right, fine veins carrying the scattered aur light across
/// them. Between them the body is a slight thread, balanced in its hover by a
/// long trailing streamer. Its feeding tube is kept coiled tight as a spring
/// beneath the cowl until a flower is reached, and a fuzzy collar behind the
/// head gathers the pollen it carries from one bloom to the next.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(0.3, 0.3, 1.6), 0.28);
    let sway = (c.animation.z * 1.1 + p.z * 0.6).sin() * 0.15 * (p.z - 1.4).max(0.0) / 3.6;
    let streamer = tapered_tail(p, vec3(0.0, 0.0, 1.4), vec3(sway, -0.25, 5.0), 0.09, 0.012);
    let mut hit = Surface::new(smooth_min(body, streamer, 0.12), 0);
    let around = p.y.atan2(p.x);
    let fuzz = (around * 24.0).sin() * (p.z * 40.0).sin() * 0.03;
    let collar = vec2(vec2(p.x, p.y).length() - 0.36, p.z + 1.45).length() - 0.15 - fuzz;
    hit.join(collar * 0.8, 4);
    // The feeding tube, coiled in the plane beneath the cowl.
    let centre = mouth(1.6, 0.28) + vec3(0.0, -0.32, 0.08);
    let q = p - centre;
    let radial = vec2(q.z, q.y).length();
    let turn = q.y.atan2(q.z) + PI;
    let spacing = 0.075 * TAU / 2.0;
    let phase = (radial - 0.04) / spacing - turn / TAU;
    let gap = ((phase + 0.5).fract_gl() - 0.5).abs() * spacing;
    let coil = (vec2(gap, q.x).length() - 0.028).max(radial - 0.3);
    hit.join(coil, 8);
    hit.join(eyes(p, 1.6, 0.28), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.25, 0.28, 0.36, 0.0),
    vec3(0.17, 0.74, 0.68),
    [vec4(7.5, 4.0, -0.5, 0.2), vec4(7.0, 3.8, 0.6, 0.2)],
)
.hinged([vec2(0.32, 0.1), vec2(0.32, 0.1)])
.styled([vec4(-1.4, 1.0, 6.0, 0.0), vec4(1.6, 1.0, 5.0, 0.0)])
// Light wings for hovering, carried on fine veins rather than bone.
.framed([0.18, 0.18])
.skinned(vec4(3.0, 0.6, 3.0, 0.15))
.flying(vec4(1.40, 0.40, 0.50, 1.20))
.marked(vec4(5.00, 4.00, 0.80, 0.38), vec3(0.95, 0.50, 0.75));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
