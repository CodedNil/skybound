use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    social: Social::Guardian,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{
    CreatureInstance, Rank, Taxon, eyes, gills, prepare_pose, smooth_min, tail, torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 41.5;

pub const TAXON: Taxon = Taxon {
    name: "Ferrispina custos",
    english_name: "Iron Spined Guardians",
    description: "Iron Spined Guardians are a nursery-guarding species of Shredder, with a robust serpentine body and conspicuous spinal blades. \
        They retain the fast banking flight and cutting attacks of their genus. \
        Their presence around nursery zones discourages approaching threats. \
        A heavier silhouette and guarded movements distinguish them from the more roaming forms of Ferrispinae. Their inherited oral tentacles pull food into the digestive crop and retain it during flight.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 6.0, 14.0, 0);
    c.opaque_bounds = vec3(6.0, 7.0, 27.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.5, 2.3, 7.0), 1.1), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.5);
    for i in 0..7 {
        let z = i as f32 * 2.0 - 5.5;
        hit.join(
            super::blade(p, vec3(0.0, 1.8, z), vec3(0.0, 5.6, z + 2.0), 0.7),
            6,
        );
    }
    hit.join(gills(p, 7.0, 2), 9);
    hit.join(eyes(p, 7.0, 1.1), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -10.0), 1.1, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.26, 0.45, 0.43, 0.0),
        vec3(0.86, 0.68, 0.28),
        [vec4(17.0, 3.0, 0.0, 0.98), vec4(11.0, 2.2, 7.0, 0.98)],
    )
}
