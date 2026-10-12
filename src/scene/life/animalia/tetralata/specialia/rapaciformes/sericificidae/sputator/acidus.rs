use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 75.0,
    lifespan_days: 1500.0,
    speed: 10.0,
    social: Social::Solitary,
    adult_mass: 0.12,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, eyes};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 24.7;

pub const TAXON: Taxon = Taxon {
    name: "Sputator acidus",
    english_name: "Spitfire Beetles",
    description: "Spitfire Beetles are small, armoured predators with specialised mouths that launch sticky acidic saliva. \
        The projected secretion can immobilise prey or corrode a threat before the animal approaches. \
        Compact wings and gripping mouth palps steady them among vegetation and Weaver structures. \
        Despite their size, their ability to attack from a distance makes them dangerous to disturb.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 5.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.7, 1.4, 2.8)), 5);
    hit.join(ellipsoid(p - vec3(0.0, 0.15, -2.2), vec3(1.2, 1.0, 1.2)), 5);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -3.20), 0.85, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(
        capsule(p, vec3(0.0, -0.3, -2.8), vec3(0.0, -0.2, -4.0), 0.3),
        8,
    );
    hit.join(eyes(p, 2.8, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.18, 0.44, 0.29, 0.0),
        vec3(0.28, 0.58, 0.62),
        [vec4(6.0, 2.2, 0.0, 0.95), vec4(4.0, 1.8, 3.0, 0.32)],
    )
}
