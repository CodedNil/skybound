use crate::scene::life::biology::{Biology, Energy};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1200.0,
    lifespan_days: 60000.0,
    energy: Energy::Lightning,
    adult_mass: 8.0,
    ..super::BIOLOGY
};

use core::f32::consts::FRAC_PI_2;

use spirv_std::glam::{Vec3, vec3};

use crate::scene::life::plantae::{leaf, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 20.0;

pub const TAXON: Taxon = Taxon {
    name: "Tempestivaga fulgurivora",
    english_name: "Stormchasers",
    description: "Stormchasers are sail-bearing woody plants common in electrically active regions. \
        They sense approaching storms, position themselves to capture lightning energy and then use their sails to move onward. \
        Broad upright membranes surround a narrow supporting body, with energy travelling through collecting veins. \
        Their arrival is a warning that storm conditions may soon follow.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(10.0, 12.0, 10.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -7.0, 0.0), vec3(0.0, 8.0, 0.0), 0.3),
        0,
    );
    shape.join(ellipsoid(p - vec3(0.0, -3.0, 0.0), vec3(1.3, 2.0, 1.3)), 3);
    for i in 0..4 {
        let local = rotate(p, i as f32 * FRAC_PI_2);
        let sail = vec3(local.x, local.z, local.y + 4.0);
        shape.join(leaf(sail, 11.0, 5.0, 1.4), 1);
        shape.join(
            capsule(local, vec3(0.0, 2.0, 0.0), vec3(4.5, 3.5, 0.0), 0.09),
            2,
        );
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.16, 0.22, 0.46),
        vec3(0.55, 0.7, 1.0),
    )
}
