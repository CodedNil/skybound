use crate::scene::life::biology::{Biology, Reproduction};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 180.0,
    lifespan_days: 5000.0,
    adult_mass: 7.0,
    reproduction: Reproduction::Fragmentation,
    altitude: (20.0, 800.0),
    ..super::BIOLOGY
};

use core::f32::consts::TAU;

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::rotate;
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 16.0;

pub const TAXON: Taxon = Taxon {
    name: "Stelloradix pendula",
    english_name: "Root-Stars",
    description: "Root-Stars have a buoyant rubbery sphere above a star-shaped cluster of thick fleshy roots. \
        The roots hang into lower aur mists and absorb nutrients from rising energy-rich air. \
        Continued exposure eventually kills and detaches them. \
        Detached roots drift upward, curl and burst into new rubbery bodies that grow their own hanging roots, making this loss part of the reproductive cycle.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 11.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p, vec3(2.7, 2.0, 2.7)), 3);
    for i in 0..8 {
        let angle = i as f32 * TAU / 8.0;
        let local = rotate(p, angle);
        let t = (-local.y / 8.0).clamp(0.0, 1.0);
        let center = vec3(0.0, -t * 8.0, 1.4 + t * 4.0 + (t * 4.0).sin() * 0.4);
        shape.join(
            ((local - center).length() - (0.65 * (1.0 - t) + 0.08)) * 0.55,
            1,
        );
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.48, 0.27, 0.17),
        vec3(0.7, 0.54, 0.3),
    )
}
