use crate::scene::life::{
    CreatureInstance,
    biology::Biology,
    capsule, ellipsoid, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3, vec4};

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

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.48, 0.55, 2.2)), 0);
    hit.distance = smooth_min(
        hit.distance,
        ellipsoid(p - vec3(0.0, 0.1, -2.0), vec3(0.65, 0.65, 0.70)),
        0.2,
    );
    let q = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        capsule(q, vec3(0.35, 0.4, -2.3), vec3(1.5, 1.8, -3.5), 0.045),
        4,
    );
    hit.join(
        ellipsoid(q - vec3(0.48, 0.25, -2.35), vec3(0.23, 0.28, 0.30)),
        3,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.25, 0.28, 0.36, 0.0),
        vec3(0.17, 0.74, 0.68),
        [vec4(7.0, 3.5, 1.0, 0.24), vec4(5.0, 2.7, 4.0, 0.28)],
    )
}
