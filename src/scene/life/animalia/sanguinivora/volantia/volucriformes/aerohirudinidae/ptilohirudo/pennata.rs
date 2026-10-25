use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    eyespots, feathers, prepare_pose, smooth_min, sucker,
    surface::{Ray, Surface, TraceResult},
    tail,
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 21.0,
    lifespan_days: 480.0,
    speed: 24.0,
    adult_mass: 0.01,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 30.4;

taxon!(
    Species,
    "Ptilohirudo pennata",
    "Feathered Leeches",
    [
        "Feathered Leeches glide along cloud margins on finely fringed wings surrounding a compact body.",
        "Their vision detects weak aur-lit contrasts, helping them find hosts hidden by mist.",
        "A quick attachment supplies a fluid meal before they release and return to cover.",
        "They favour broken cloud edges where passing animals are visible without requiring a long flight through dry air.",
        "They favour humid cloud pockets on the warmer margins of the equatorial Highlands, avoiding both frozen Shard Peaks and the strongest poleward winds.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 5.0, 0);
    c.opaque_bounds = vec3(4.0, 2.0, 10.0).extend(0.0);
}

/// A feathered flying leech: a ringed body running into a feathered tail,
/// a gripping cup at the mouth and eyespots along the head.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = annulated(
        p,
        vec3(0.0, 0.0, -2.7),
        vec3(0.0, 0.05, 2.8),
        vec3(0.34, 0.62, 0.3),
        14.0,
        0.06,
        0.8,
    );
    let mut hit = Surface::new(smooth_min(body, tail(p, c) * 1.6, 0.3), 0);
    hit.join(feathers(p, c, 2.0), 1);
    hit.join(
        sucker(
            p,
            vec3(0.0, -0.05, -2.8),
            vec3(0.0, -0.3, -1.0).normalize(),
            0.3,
        ),
        8,
    );
    hit.join(eyespots(p, vec3(0.16, 0.2, -2.45), 0.25, 2, 0.05), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.25, 0.32, 0.42, 0.0),
    vec3(0.69, 0.73, 0.80),
    [vec4(9.0, 3.0, -0.6, 0.98), Vec4::ZERO],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
