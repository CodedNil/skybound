use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, legs, prepare_pose};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Lucinani",
    english_name: "Glowgnomes",
    description: "Glowgnomes are tiny luminous sap feeders that live in burrows within Whisperwood. \
        Their bioluminescence helps attract pollinators, while their wings collectively pull the tree toward favourable climates or away from danger. \
        Pheromones carried through the tree's six sap veins coordinate groups on different sides. \
        Although descended from disguised parasites, they now receive food and shelter in exchange for useful light and active movement.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
    radius: 16.0,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 2.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 1.5, 5.0, 0);
    c.opaque_bounds = vec3(4.0, 4.0, 3.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 1.6, 1.0)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 1.3, 0.0), vec3(0.85, 0.9, 0.85)), 9);
    hit.join(legs(p, vec3(0.7, 1.6, 1.0), 2), 4);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.55, 0.32, 0.15, 0.0),
        vec3(1.0, 0.72, 0.20),
        [vec4(3.0, 1.3, 0.0, 0.32)],
    )
}
