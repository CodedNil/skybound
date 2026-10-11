use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Chimney,
        ..super::BIOLOGY.organs
    },
    adult_mass: 60.0,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Susurrium",
    english_name: "Whisperwood",
    description: "Whisperwood is a rare tree with a huge upright hollow trunk, wider at the base, that channels air for lift and stability. \
        Bamboo-like ridges strengthen the cylinder, and hollow branches end in small openings that whistle at different pitches. \
        Delicate fan leaves spread around the branches. \
        Glowgnomes live in the wood, receiving sap and shelter while providing light and active movement. \
        Six vertical sap veins carry pheromone requests to groups on different sides, allowing the tree to seek favourable climates or evade danger.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 32.0,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(13.0, 22.0, 13.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let radial = p.xz().length();
    let radius = 1.8 - p.y * 0.04;
    let rings = (p.y * 2.5).cos().smoothstep(0.75, 0.95) * 0.15;
    let trunk = ((radial - radius - rings).abs() - 0.22).max(p.y.abs() - 17.0) * 0.75;
    let mut shape = Surface::new(trunk, 0);
    for i in 0..12 {
        let angle = i as f32 * 2.4;
        let start = vec3(
            angle.cos() * radius,
            i as f32 * 2.1 - 9.0,
            angle.sin() * radius,
        );
        let end = start + vec3(angle.cos() * 6.0, 2.0, angle.sin() * 6.0);
        let branch = capsule(p, start, end, 0.25).max(-capsule(p, start, end, 0.14));
        shape.join(branch, 0);
        let mut local = rotate(p - end, angle);
        local.z += 1.0;
        shape.join(rosette(local, 5, 3.6, 1.3, 0.6), 1);
        shape.join(ellipsoid(p - start, Vec3::splat(0.14)), 2);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.15, 0.44, 0.3),
        vec3(0.37, 0.85, 0.56),
    )
}
