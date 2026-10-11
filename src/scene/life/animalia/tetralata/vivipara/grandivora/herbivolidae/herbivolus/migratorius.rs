use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 12.0,
    adult_mass: 35.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{
    CreatureInstance, Rank, Taxon, capsule, eyes, feathers, prepare_pose, smooth_min, tail, torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 49.8;

pub const TAXON: Taxon = Taxon {
    name: "Herbivolus migratorius",
    english_name: "Grazers",
    description: "Grazers are large, slow-moving herbivores with broad feathered wings and long, soft tongue-like feeding appendages. \
        Their soft feeding appendages gather larger algae as they sweep through dense concentrations of food. \
        Twice a year they return to the Mineral Fields in enormous migratory herds. \
        Their steady flight and social gatherings make them a conspicuous part of the aerial landscape. Their inherited oral tentacles pull food into the digestive crop and retain it during flight.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 8.0, 14.0, 0);
    c.opaque_bounds = vec3(7.0, 6.0, 22.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(4.8, 4.2, 9.0), 1.4), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.5);
    hit.join(feathers(p, c, 4.0), 1);
    hit.join(
        capsule(p, vec3(0.0, -1.0, -11.0), vec3(0.0, -3.0, -17.0), 0.45),
        8,
    );
    hit.join(eyes(p, 9.0, 1.4), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -13.0), 2.3, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.68, 0.54, 0.31, 0.0),
        vec3(0.87, 0.75, 0.54),
        [vec4(20.0, 6.0, 0.0, 0.98), vec4(14.0, 4.0, 10.0, 0.98)],
    )
}
