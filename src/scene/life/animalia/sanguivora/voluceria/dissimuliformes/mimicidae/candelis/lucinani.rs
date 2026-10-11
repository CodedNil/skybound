use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Grazer,
    social: Social::Mutualist,
    adult_mass: 0.002,
    speed: 1.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, prepare_pose};
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
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 1.5, 5.0, 0);
    c.opaque_bounds = vec3(4.0, 4.0, 3.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 1.6, 1.0)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 1.3, 0.0), vec3(0.85, 0.9, 0.85)), 9);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -1.40), 0.35, BIOLOGY.oral_pairs),
        8,
    );
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
