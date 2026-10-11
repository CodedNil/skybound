use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Cleaner,
    social: Social::Mutualist,
    speed: 8.0,
    adult_mass: 0.08,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes, prepare_pose};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Purgatoridae",
    english_name: "Sweepers",
    description: "Sweepers are small, bloated animals that flit around larger creatures using specialised mouth palps to remove parasites and debris. \
        Their rounded bodies and repeated close approaches identify them as persistent grooming companions. \
        They obtain food from this cleaning while improving the health of their hosts. \
        Hosts tolerate their repeated visits in exchange for grooming and parasite removal.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 21.6,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.5, 2.0, 0);
    c.opaque_bounds = vec3(6.0, 5.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(2.3, 2.0, 2.7)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 0.4, -2.5), vec3(1.0, 0.9, 0.9)), 0);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -3.10), 1.15, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 2.7, 0.5), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.68, 0.74, 0.51, 0.0),
        vec3(0.35, 0.59, 0.43),
        [vec4(6.0, 2.5, 0.0, 0.4), vec4(4.0, 1.8, 3.0, 0.4)],
    )
}
