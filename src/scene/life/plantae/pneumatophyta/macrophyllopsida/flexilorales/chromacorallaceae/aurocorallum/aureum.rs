use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 900.0,
    lifespan_days: 30000.0,
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    adult_mass: 1.0,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 13.0;

pub const TAXON: Taxon = Taxon {
    name: "Aurocorallum aureum",
    english_name: "Golden Coral",
    description: "Golden Coral grows as large gold-hued branching colonies with layered, frilled collecting surfaces. \
        It attaches primarily to large slow-moving animals and occasionally to plants. \
        A host supplies transport and support while the colony gathers ambient aur. \
        Its warm metallic-looking colours distinguish it from the more varied palette of Colour Coral.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -2.0, 0.0), vec3(1.8, 0.6, 1.8)), 3);
    for i in 0..9 {
        let angle = i as f32 * 2.4;
        let start = vec3(angle.cos(), -2.0, angle.sin());
        let end = vec3(
            angle.cos() * 3.5,
            1.5 + (angle * 2.0).sin(),
            angle.sin() * 3.5,
        );
        shape.join(capsule(p, start, end, 0.22), 3);
        for j in 0..3 {
            let tip = end
                + vec3(
                    (angle + j as f32).cos() * 1.2,
                    j as f32 * 0.7,
                    (angle + j as f32).sin() * 1.2,
                );
            shape.join(capsule(p, end, tip, 0.12), 3);
            shape.join(rosette(rotate(p - tip, angle), 4, 1.0, 0.4, 0.5), 1);
        }
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, vec3(0.75, 0.42, 0.07), vec3(1.0, 0.7, 0.2))
}
