pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance,
    animalia::tetralata::oral_tentacles,
    biology::Biology,
    ellipsoid, eyes,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3, vec4};

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
        "Weavers hold themselves between kelp fronds on four wings while mouth tentacles draw out and fasten silk.",
        "Loose catching strands hang across feeding approaches, while stronger braided lines keep neighbouring plants from drifting beyond reach.",
        "They read changes in tension to distinguish grazing visitors from moving anchors, then reel captured prey toward the sheltered centre.",
        "Lines are loosened or replaced as currents shift, allowing the structure to breathe with a Jungle rather than resist it as a rigid net.",
        "Nests, collected bright fragments and the work of Næ crafters accumulate along old anchor corridors, turning a feeding apparatus into a persistent aerial neighbourhood.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 6.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.3, 1.0, 2.4)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 0.0, -2.2), vec3(1.0, 0.8, 1.0)), 0);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -2.80), 0.65, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(ellipsoid(p - vec3(0.0, -0.6, 2.0), vec3(0.65, 0.6, 0.8)), 7);
    hit.join(eyes(p, 2.4, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.35, 0.24, 0.44, 0.0),
        vec3(0.83, 0.58, 0.30),
        [vec4(6.0, 2.0, 0.0, 0.27), vec4(4.0, 1.5, 3.0, 0.27)],
    )
}
