use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1800.0,
    lifespan_days: 60000.0,
    ..super::BIOLOGY
};
use crate::scene::life::animalia::tetralata::oral_tentacles;

use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 46.8;

pub const TAXON: Taxon = Taxon {
    name: "Lithosoma grave",
    english_name: "Grounders",
    description: "Grounders are massive, mineral-crusted herbivores with rocky hides and broad lifting surfaces. \
        Their dense bodies drift slowly downward through the lower mid-level currents. \
        They inhabit lower mid-level currents and feed on dense Root-Stars and materials around mineral geysers. \
        Mineral armour covers their substantial bodies, and their movements are slow and deliberate.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(12.0, 10.0, 13.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(8.0, 6.0, 10.0)), 5);
    for i in 0..7 {
        let f = i as f32;
        hit.join(
            ellipsoid(
                p - vec3((f * 2.4).sin() * 5.0, 3.7 + (f * 1.7).cos(), f * 2.5 - 7.0),
                vec3(3.4, 3.7, 3.0),
            ),
            5,
        );
    }
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -10.40), 4.00, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 10.0, 0.9), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.24, 0.24, 0.29, 0.0),
        vec3(0.46, 0.32, 0.56),
        [vec4(20.0, 5.5, 0.0, 0.98), vec4(14.0, 4.0, 11.0, 0.98)],
    )
}
