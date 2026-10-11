use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, prepare_pose, ribbons};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Fluitans",
    english_name: "Floating Ones",
    description: "Floating Ones are born among aerial kelp mats, with green-brown colours and thin lateral folds that resemble surrounding vegetation. \
        Their soft trailing folds sway with the kelp, concealing the body among its fronds. \
        They wait for a feeding creature to brush past, attach with a sucker and remain on that host permanently. \
        Dispersal is carried by the wind and the movement of their nursery mats.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 26.0,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::drift(senses, 0.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 9.0, 2);
    c.opaque_bounds = vec3(3.0, 3.0, 18.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 0.55, 4.0)), 0);
    hit.join(ribbons(p, c, 2), 4);
    for i in 0..6 {
        let z = i as f32 * 1.2 - 3.0;
        hit.join(
            ellipsoid(vec3(p.x.abs() - 1.1, p.y, p.z - z), vec3(1.4, 0.12, 0.6)),
            4,
        );
    }
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.24, 0.39, 0.19, 0.0),
        vec3(0.53, 0.61, 0.25),
        [],
    )
}
