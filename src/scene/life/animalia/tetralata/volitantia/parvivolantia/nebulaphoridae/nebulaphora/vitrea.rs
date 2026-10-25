use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::curved_tube,
    biology::Biology,
    eyes, mouth,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
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
        "Mistings spread transparent, coloured wings around a soft central body, catching scattered aur light along fine supporting veins.",
        "Each wing pair adjusts separately as they hover beside a swaying Bloom.",
        "Flexible mouth appendages reach the nectar without forcing the animal to land on the petals.",
        "Pollen clings during contact and travels with the next visit, linking flowers across open gaps in the Gardens.",
        "They favour sheltered flowers along humid Swampland margins, where rainfall provides moisture without requiring an approach to the lethal geyser jets.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(2.0, 2.1, 4.0).extend(0.0);
}

/// A soft, slender nectar-feeder between its wings: a furred thorax, a
/// long ringed abdomen, and a small head with neat eyes and curling
/// feelers.
/// A broad-winged nectar-feeder: a slight body beneath its wings, and in
/// place of the ancestral tentacle pair a single long feeding tube, held
/// curled beneath the cowl, that reaches into a bloom without the animal
/// ever settling on it.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.45, 0.45, 1.9), 0.32), 0);
    let lip = mouth(1.9, 0.32);
    let reach = (c.animation.z * 0.9).sin() * 0.15;
    hit.join(
        curved_tube(
            p,
            lip,
            lip + vec3(0.0, -0.7, -0.3),
            lip + vec3(0.0, -0.35 + reach, -1.15 - reach),
            0.06,
        ),
        8,
    );
    hit.join(eyes(p, 1.9, 0.32), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.25, 0.28, 0.36, 0.0),
    vec3(0.17, 0.74, 0.68),
    [vec4(7.0, 3.5, -0.6, 0.24), vec4(5.0, 2.7, 0.5, 0.28)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
