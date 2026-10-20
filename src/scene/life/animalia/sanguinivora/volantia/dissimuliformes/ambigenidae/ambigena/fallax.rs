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
    lifespan_days: 2400.0,
    speed: 8.0,
    adult_mass: 0.04,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 22.6;

taxon!(
    Species,
    "Ambigena fallax",
    "Double Natured",
    [
        "Double Natured animals resemble Sweepers, with swollen bodies and mobile mouth fringes used during close grooming visits.",
        "They remove debris and apply antiseptic secretions while taking small amounts of host fluid through the inherited feeding sucker.",
        "Genuine cleaning reduces the irritation of feeding and gives the host a reason to tolerate their presence.",
        "Frequent short visits are safer than remaining attached where a host can repeatedly inspect the same patch of skin.",
        "Their deceptive resemblance arose in a winged parasite lineage distinct from the four-winged cleaners they imitate.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 5.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.6, 1.5, 2.8)), 0);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -3.20), 0.80, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(
        ellipsoid(p - vec3(0.0, -0.5, -2.0), vec3(0.7, 0.45, 1.0)),
        8,
    );
    hit.join(eyes(p, 2.8, 0.55), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.67, 0.72, 0.46, 0.0),
        vec3(0.57, 0.38, 0.29),
        [vec4(6.0, 2.3, 0.0, 0.35)],
    )
}
