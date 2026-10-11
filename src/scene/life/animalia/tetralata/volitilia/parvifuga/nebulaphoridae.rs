use crate::scene::life::biology::{Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Nectar,
    adult_mass: 0.1,
    speed: 5.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, capsule, ellipsoid, prepare_pose, smooth_min,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Nebulaphoridae",
    english_name: "Mistings",
    description: "Mistings are large, docile moth-like pollinators with two pairs of thin, transparent wings. \
        Veins and coloured membranes catch light like stained glass, surrounding a soft central body. \
        They carry pollen between widely separated plant clusters and are especially attracted to Blooms. \
        Their slow, gentle flight carries them between flowering plants and distant pollen sources.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 19.8,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 2.4, 3.0, 2);
    c.opaque_bounds = vec3(2.0, 2.1, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.48, 0.55, 2.2)), 0);
    hit.distance = smooth_min(
        hit.distance,
        ellipsoid(p - vec3(0.0, 0.1, -2.0), vec3(0.65, 0.65, 0.70)),
        0.2,
    );
    let q = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        capsule(q, vec3(0.35, 0.4, -2.3), vec3(1.5, 1.8, -3.5), 0.045),
        4,
    );
    hit.join(
        ellipsoid(q - vec3(0.48, 0.25, -2.35), vec3(0.23, 0.28, 0.30)),
        3,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.25, 0.28, 0.36, 0.0),
        vec3(0.17, 0.74, 0.68),
        [vec4(7.0, 3.5, 1.0, 0.24), vec4(5.0, 2.7, 4.0, 0.28)],
    )
}
