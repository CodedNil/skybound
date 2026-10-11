use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.001,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{
    Material, Ray, Shell, Surface, TraceResult, trace_material, visit_shell_material,
};

pub const TAXON: Taxon = Taxon {
    name: "Auracoccus",
    english_name: "Algae",
    description: "Algae are diverse microscopic plants suspended as individual particles in the updraft. \
        Their small rounded cells synthesise energy from collected aur and multiply rapidly through asexual division. \
        Budding and dividing cells form the smallest visible growth stages. \
        They are a primary food source for filter feeders and accumulate on sticky animals, membranes and plant surfaces.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 0.00035,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(0.0002, 0.0002, 0.0002).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let p = p * 10000.0;
    let local = p + vec3((p.y * 7.0).sin(), (p.z * 8.0).sin(), (p.x * 6.0).sin()) * 0.018;
    let mut shape = Surface::new(ellipsoid(local, vec3(0.85, 0.72, 0.85)), 1);
    shape.join(
        ellipsoid(local - vec3(0.75, 0.3, 0.2), vec3(0.6, 0.55, 0.6)),
        1,
    );
    shape.join(ellipsoid(p, Vec3::splat(0.35)), 2);
    shape.distance *= 0.0001;
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::opaque(trace_material(
        ray,
        c,
        |p, _| Surface::new(ellipsoid(p, Vec3::splat(0.000_035)), 0),
        |_, _| Material::skin(vec3(0.08, 0.4, 0.12), 0.35),
    ));
    for (center, radii) in [
        (Vec3::ZERO, vec3(0.85, 0.72, 0.85)),
        (vec3(0.75, 0.3, 0.2), vec3(0.6, 0.55, 0.6)),
    ] {
        visit_shell_material(
            ray,
            c,
            Shell {
                center: center * 0.0001,
                radii: radii * 0.0001,
                bottom: -1.0,
                scallop: 0.0,
            },
            |_, cosine| {
                let mut cell = Material::skin(vec3(0.16, 0.56, 0.28), 0.2);
                cell.ior = 1.36;
                cell.translucency = 0.5;
                cell.transmission = (-vec3(0.3, 0.06, 0.2) / cosine.max(0.15)).exp() * 0.85;
                cell
            },
            |hit| result.add(hit),
        );
    }
    result
}
