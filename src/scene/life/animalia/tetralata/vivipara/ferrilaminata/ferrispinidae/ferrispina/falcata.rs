pub use super::BIOLOGY;
use super::blade;
use crate::scene::life::animalia::tetralata::oral_tentacles;

use spirv_std::glam::{Vec3, vec3, vec4};

pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, eyes, prepare_pose, smooth_min, tail, torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 41.5;

pub const TAXON: Taxon = Taxon {
    name: "Ferrispina falcata",
    english_name: "Shredders",
    description: "Shredders are agile, serpentine apex predators with razor-sharp blades along their spines. \
        Long flexible bodies and two pairs of wings let them sweep past prey in tight, cutting arcs. \
        The spinal blades cut through tough hides during a sweeping pass. \
        Younger animals often hunt in coordinated packs, separating prey from protective groups before striking. Their inherited oral tentacles pull food into the digestive crop and retain it during flight.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 14.0, 0);
    c.opaque_bounds = vec3(5.0, 6.0, 29.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.7, 1.5, 8.0), 0.8), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    for i in 0..9 {
        let z = i as f32 * 1.6 - 6.0;
        hit.join(blade(p, vec3(0.0, 1.2, z), vec3(0.0, 4.5, z + 2.0), 0.5), 6);
    }
    hit.join(eyes(p, 8.0, 0.8), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -10.0), 1.0, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.19, 0.22, 0.31, 0.0),
        vec3(0.66, 0.19, 0.14),
        [vec4(15.0, 2.8, 0.0, 0.98), vec4(10.0, 2.0, 8.0, 0.98)],
    )
}
