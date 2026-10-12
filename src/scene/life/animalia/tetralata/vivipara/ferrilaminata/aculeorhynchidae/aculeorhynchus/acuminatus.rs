use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 900.0,
    lifespan_days: 18000.0,
    speed: 75.0,
    social: Social::Solitary,
    adult_mass: 14.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes, tapered_tail, torso};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 41.5;

pub const TAXON: Taxon = Taxon {
    name: "Aculeorhynchus acuminatus",
    english_name: "Needlesharks",
    description: "Needlesharks carry a huge needle-like spike on the head of a streamlined four-winged body. \
        They approach from above and align the whole animal with the intended strike. \
        A sudden swoop drives the spike toward vital organs before prey has time to respond. \
        Their pointed silhouette and steep attack angle distinguish them from the blade-bearing hunters nearby in the tree. Their inherited oral tentacles pull food into the digestive crop and retain it during flight.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

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
