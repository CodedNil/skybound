use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    annulated,
    biology::Biology,
    rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 400.0,
    lifespan_days: 12000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 26.8;

taxon!(
    Species,
    "Ramiphaga corticivora",
    "Driftmunchers",
    [
        "Driftmunchers brace compact wings around an armoured body while crushing the bark and thorns of drifting Brambles.",
        "Reinforced mandibles work beside the inherited mouth appendages, which hold a branch without functioning as walking limbs.",
        "They strip outer growth in small patches so that feeding does not immediately destroy the framework supporting them.",
        "Protective tissues tolerate brief contact with charged spines, but the soft underside remains vulnerable during a careless turn.",
        "Their feeding opens resistant woody nutrients to smaller scavengers and leaves sheltered gaps used by other inhabitants of the tangle.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 7.0).extend(0.0);
    rig_tentacles(c, vec3(0.0, -0.6, -4.2), 1.35, BIOLOGY.oral_pairs);
}

/// A bark-eater armoured in banded plates, with a blunt head, small deep
/// eyes and a pair of heavy gouging tusks.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut body = annulated(
        p,
        vec3(0.0, 0.0, -2.6),
        vec3(0.0, 0.3, 3.8),
        vec3(2.0, 2.7, 0.9),
        5.0,
        0.12,
        0.6,
    );
    for i in 0..5 {
        let z = i as f32 * 1.25 - 2.6;
        let plate = rounded(p - vec3(0.0, 0.75, z), vec3(2.75, 1.0, 0.72));
        body = smooth_min(body, plate, 0.12);
    }
    let head = rounded(p - vec3(0.0, -0.1, -3.2), vec3(1.6, 1.25, 1.2));
    let mut hit = Surface::new(smooth_min(body, head, 0.4), 5);
    let q = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        rounded(q - vec3(1.05, 0.45, -3.95), vec3(0.24, 0.22, 0.26)),
        3,
    );
    hit.join(
        tapered_tail(q, vec3(0.9, -0.4, -3.6), vec3(0.6, -0.75, -6.0), 0.42, 0.08),
        6,
    );
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.6, -4.2), 1.35, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.34, 0.22, 0.13, 0.0),
    vec3(0.56, 0.36, 0.17),
    [vec4(8.0, 3.0, -1.0, 0.9), vec4(5.0, 2.0, 1.3, 0.36)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
