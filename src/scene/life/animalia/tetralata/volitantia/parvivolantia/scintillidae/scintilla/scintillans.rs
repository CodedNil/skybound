use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    eyes, rounded,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 3.0,
    lifespan_days: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 19.8;

taxon!(
    Species,
    "Scintilla scintillans",
    "Sparkies",
    [
        "Sparkies appear as moving points of intense colour as vast swarms gather algae and suspended seeds.",
        "Two pairs of quick wings let each animal adjust to its neighbours and to small changes in the feeding current.",
        "Their light preserves contact through dim aur phases, but also draws Shimmerays and other small-prey hunters.",
        "Fast growth allows dense swarms to appear around a brief food pulse and disperse before the current empties.",
        "Such gatherings trouble larger animals and Næ settlements while supplying an essential link between microscopic plant life and aerial predators.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.2, 1.2, 2.4).extend(0.0);
}

/// A tiny flyer like a firefly: a small head with domed eyes and feathery
/// antennae, a rounded thorax, and a ringed abdomen whose last segments
/// hold the light.
/// A tiny swarming algae-eater that carries its own light. The lamps sit
/// along the top of its back and at the tail, facing up and back: a swarm
/// keeps together by them through the dim phase, while a hunter rising
/// from below sees only a dark belly against the dark sky.
pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.32, 0.3, 1.2), 0.2), 0);
    hit.join(rounded(p - vec3(0.0, 0.1, 1.05), vec3(0.17, 0.13, 0.3)), 9);
    for i in 0..3 {
        let z = i as f32 * 0.32 - 0.6;
        hit.join((p - vec3(0.0, 0.31 - z * 0.04, z)).length() - 0.055, 9);
    }
    hit.join(eyes(p, 1.2, 0.2), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.22, 0.71, 0.79, 0.65),
    vec3(0.76, 0.98, 0.42),
    [vec4(3.0, 1.3, -0.3, 0.24), vec4(2.3, 1.0, 0.3, 0.27)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
