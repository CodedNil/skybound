use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes, legs, prepare_pose};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Ambigena",
    english_name: "Double Natured",
    description: "Double Natured animals imitate the swollen bodies and specialised limbs of Sweepers. \
        They remove debris and provide an antiseptic cleaning benefit while slowly drawing fluids from the same host. \
        Their appearance and careful movements encourage hosts to tolerate close contact. \
        Hosts often tolerate their careful grooming movements, allowing prolonged access to a food source.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 22.6,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 12.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.5, 2.0, 0);
    c.opaque_bounds = vec3(5.0, 5.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.6, 1.5, 2.8)), 0);
    hit.join(legs(p, vec3(1.6, 1.5, 2.8), 2), 4);
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
