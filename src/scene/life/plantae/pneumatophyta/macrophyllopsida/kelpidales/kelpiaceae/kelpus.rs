use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::plantae::{frond, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Kelpus",
    english_name: "Kelp",
    description: "Kelp grows as long ribbons of fibrous plant material with small gas bladders spaced along its length. \
        Broad fronds sway around the buoyant strand and collect aur over a large surface. \
        Balanced lift lets enormous strands spread into expansive, nearly horizontal forests. \
        These forests feed grazers and Wise Ones while providing attachment sites and sheltered routes through the sky.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 25.0,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::seek_aur(senses, 0.3)
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 5.0, 17.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let center = vec3(
        (p.z * 0.28 + c.animation.z * 0.3).sin() * 0.8,
        (p.z * 0.4).sin() * 0.6,
        p.z.clamp(-13.0, 13.0),
    );
    let mut shape = Surface::new((p - center).length() * 0.7 - 0.12, 0);
    for i in -4..=4 {
        let z = i as f32 * 2.8;
        let root = vec3(
            (z * 0.28 + c.animation.z * 0.3).sin() * 0.8,
            (z * 0.4).sin() * 0.6,
            z,
        );
        shape.join(
            ellipsoid(p - root - vec3(0.0, 0.6, 0.0), vec3(0.45, 0.7, 0.45)),
            3,
        );
        for side in [-1.0, 1.0] {
            let local = rotate(p - root, side * 1.1);
            shape.join(
                frond(local, 5.0, 1.0, 0.5 + 0.2 * (z + c.animation.z).sin(), 4.0),
                1,
            );
        }
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.22, 0.46, 0.08),
        vec3(0.57, 0.69, 0.15),
    )
}
