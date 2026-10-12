use crate::scene::life::biology::{Biology, Reproduction, Social};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 2.0,
    lifespan_days: 120.0,
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    reproduction: Reproduction::Fragmentation,
    social: Social::Mutualist,
    adult_mass: 0.03,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 7.0;

pub const TAXON: Taxon = Taxon {
    name: "Luminiflora symbiotica",
    english_name: "Lumaweed",
    description: "Lumaweed is a widespread, rapidly growing bioluminescent plant that forms patches on larger lifeforms. \
        It uses ambient aur and produces a soft pervasive glow, while its edible tissue feeds many small animals. \
        Fragmentation spreads new colonies between surfaces. \
        Animal hosts receive some of the nutrients it produces, but plant hosts such as kelp experience the association as parasitism.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 3.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -0.4, 0.0), vec3(2.8, 0.3, 2.8)), 1);
    for i in 0..15 {
        let angle = i as f32 * 2.39996;
        let radius = (i as f32 / 15.0).sqrt() * 2.4;
        let root = vec3(angle.cos() * radius, 0.0, angle.sin() * radius);
        let tip = root
            + vec3(
                0.18 * angle.sin(),
                0.5 + (angle * 3.0).sin().abs(),
                0.2 * angle.cos(),
            );
        shape.join(capsule(p, root, tip, 0.07), 1);
        shape.join(ellipsoid(p - tip, vec3(0.18, 0.28, 0.18)), 2);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.09, 0.4, 0.22),
        vec3(0.15, 0.95, 0.8),
    )
}
