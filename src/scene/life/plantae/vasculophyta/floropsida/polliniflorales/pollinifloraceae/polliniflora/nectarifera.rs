use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 35.0,
    lifespan_days: 700.0,
    adult_mass: 0.5,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 11.0;

pub const TAXON: Taxon = Taxon {
    name: "Polliniflora nectarifera",
    english_name: "Blooms",
    description: "Blooms are delicate flowering plants with layered petals and reservoirs of sweet, aur-rich nectar. \
        Their flowers attract pollinators that carry pollen between widely separated individuals. \
        Stored aur gives the tissues healing value for Næ as well as food value for animals. \
        A buoyant supporting body and flexible stem hold the flower in the current.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 7.0, 6.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let stem = capsule(p, vec3(0.0, -1.5, 0.0), vec3(0.0, 0.3, 0.0), 0.18);
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -1.5, 0.0), vec3(1.2, 1.0, 1.2)), 3);
    shape.join(stem, 1);
    shape.join(rosette(p, 8, 4.3, 1.6, 1.4), 1);
    shape.join(
        rosette(rotate(p - vec3(0.0, 0.25, 0.0), 0.4), 6, 2.7, 1.1, 1.5),
        1,
    );
    shape.join(
        ellipsoid(p - vec3(0.0, 0.9, 0.0), vec3(0.85, 0.65, 0.85)),
        2,
    );
    for i in 0..7 {
        let angle = i as f32 * 0.897;
        let tip = vec3(angle.cos() * 0.6, 1.8, angle.sin() * 0.6);
        shape.join(capsule(p, vec3(0.0, 0.7, 0.0), tip, 0.04), 3);
        shape.join(ellipsoid(p - tip, Vec3::splat(0.13)), 2);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.82, 0.16, 0.43),
        vec3(1.0, 0.68, 0.15),
    )
}
