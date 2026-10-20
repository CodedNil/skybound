use crate::scene::life::{
    CreatureInstance,
    biology::{Biology, Reproduction},
    ellipsoid,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1.0,
    lifespan_days: 90.0,
    adult_mass: 0.001,
    reproduction: Reproduction::Division,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 17.0;

taxon!(
    Species,
    "Haematozoon invasor",
    "Blood Dwellers",
    [
        "Blood Dwellers spend most of their lives dividing within the fluid circulation of a host.",
        "Their small, soft bodies obtain nutrients directly from that enclosed environment rather than maintaining an exposed feeding grip.",
        "Dispersal growths develop near the host's surface and eventually release compact juveniles into the air.",
        "These juveniles have only a brief opportunity to enter another host before their reserves are exhausted.",
        "Crowded feeding grounds and close contact between animals therefore shape their spread more strongly than the long-distance wind that carries them.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.0, 1.0, 3.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.55, 0.55, 2.3)), 0);
    for i in 0..7 {
        let z = i as f32 * 0.55 - 1.6;
        hit.join(ellipsoid(p - vec3(0.4, 0.25, z), vec3(0.3, 0.35, 0.35)), 2);
    }
    hit.join(ellipsoid(p - vec3(0.0, 0.0, -2.1), vec3(0.45, 0.4, 0.5)), 8);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.54, 0.13, 0.22, 0.0),
        vec3(0.87, 0.43, 0.46),
        [],
    )
}
