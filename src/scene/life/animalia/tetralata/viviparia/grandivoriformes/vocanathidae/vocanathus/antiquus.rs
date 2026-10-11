use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, eyes, prepare_pose, ribbons, tapered_tail, torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Vocanathus antiquus",
    english_name: "Old Wise Ones",
    description: "Old Wise Ones are exceptionally rare and huge, with enormous white fur growths resembling flowing beards. \
        Their gentle movements and expansive wings give them a slow, stately silhouette. \
        They consume only Laevensails and kelp, browsing through large floating plant communities. \
        They belong to an intelligent live-bearing lineage noted for rich vocal communication.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
    radius: 55.6,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 8.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 10.0, 17.0, 5);
    c.opaque_bounds = vec3(10.0, 12.0, 27.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(6.0, 6.0, 12.0), 1.8), 0);
    hit.join(eyes(p, 12.0, 1.8), 3);
    for i in -3..=3 {
        let x = i as f32 * 0.7;
        hit.join(
            tapered_tail(
                p,
                vec3(x, -1.0, -13.0),
                vec3(x * 1.5, -10.0 + x.abs(), -7.0),
                0.65,
                0.12,
            ),
            7,
        );
    }
    hit.join(ribbons(p, c, 5), 7);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.75, 0.73, 0.64, 0.0),
        vec3(0.94, 0.92, 0.83),
        [vec4(26.0, 8.0, 0.0, 0.98), vec4(18.0, 6.0, 12.0, 0.98)],
    )
}
