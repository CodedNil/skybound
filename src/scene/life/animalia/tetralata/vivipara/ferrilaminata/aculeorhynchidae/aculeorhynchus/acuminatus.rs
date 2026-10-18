use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid, eyes, tapered_tail, torso};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 900.0,
    lifespan_days: 18000.0,
    speed: 75.0,
    social: Social::Solitary,
    adult_mass: 14.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Aculeorhynchus acuminatus",
    "Needlesharks",
    [
        "Needlesharks carry a long rigid head spike ahead of a streamlined body and two pairs of swept wings.",
        "They approach from higher air, using the updraft to recover altitude between steep, carefully aligned strikes.",
        "The spike concentrates force along the body axis, favouring a single puncture over a prolonged struggle.",
        "Mouth tentacles secure the wounded prey while the wings turn the animal clear of a falling or twisting target.",
        "Open feeding routes suit their attack better than tangled Jungles, where a missed strike could drive the spike into living wood.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 5.0, 19.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.8, 1.7, 6.0), 0.8), 0);
    hit.join(
        tapered_tail(p, vec3(0.0, 0.4, -7.8), vec3(0.0, 0.35, -18.0), 0.7, 0.025),
        6,
    );
    hit.join(ellipsoid(p - vec3(0.0, 2.0, 1.0), vec3(0.18, 1.7, 4.5)), 5);
    hit.join(eyes(p, 6.0, 0.8), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -8.0), 0.8, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.16, 0.34, 0.43, 0.0),
        vec3(0.63, 0.71, 0.73),
        [vec4(15.0, 3.0, 0.0, 0.98), vec4(9.0, 2.0, 7.0, 0.98)],
    )
}
