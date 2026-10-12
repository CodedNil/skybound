use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 400.0,
    lifespan_days: 12000.0,
    ..super::BIOLOGY
};
use crate::scene::life::animalia::tetralata::oral_tentacles;

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, tapered_tail};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 26.8;

pub const TAXON: Taxon = Taxon {
    name: "Ramiphaga corticivora",
    english_name: "Driftmunchers",
    description: "Driftmunchers are slow, armoured herbivores with strong crushing mandibles and compact supporting wings. \
        They feed on Bramble bark and thorns, carefully stripping the outer layers of each sphere. \
        Their sturdy bodies tolerate contact with woody, defensive vegetation. \
        By processing material that softer grazers cannot eat, they return nutrients locked in tough plant tissues to the wider ecosystem.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(2.7, 1.6, 3.7)), 5);
    for i in 0..5 {
        hit.join(
            ellipsoid(
                p - vec3(0.0, 0.5, i as f32 * 1.25 - 2.8),
                vec3(2.8, 1.3, 0.85),
            ),
            5,
        );
    }
    let q = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        tapered_tail(q, vec3(1.2, 0.0, -3.0), vec3(0.65, -0.2, -6.0), 0.65, 0.14),
        6,
    );
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -4.10), 1.35, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.34, 0.22, 0.13, 0.0),
        vec3(0.56, 0.36, 0.17),
        [vec4(8.0, 3.0, 0.0, 0.9), vec4(5.0, 2.0, 4.0, 0.36)],
    )
}
