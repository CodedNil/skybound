use crate::scene::life::biology::{Biology, Reproduction};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 180.0,
    lifespan_days: 5000.0,
    adult_mass: 7.0,
    reproduction: Reproduction::Fragmentation,
    altitude: (50.0, 800.0),
    ..super::BIOLOGY
};

use core::f32::consts::TAU;

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::rotate;
use crate::scene::life::{CreatureInstance, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 16.0;

taxon!(
    Species,
    "Stelloradix pendula",
    "Root-Stars",
    [
        "Root-Stars float on rubbery crowns while thick fleshy roots radiate downward into nutrient-rich lower air.",
        "Vascular channels lift absorbed moisture and minerals from the roots into the body.",
        "Roots reaching the upper Veiland mists suffer progressive surface damage and are shed before exposure spreads into the crown.",
        "Surviving inner tissue in a detached root curls, forms a buoyant body and rises to begin another plant.",
        "Grounders feed on these rich root structures, linking a dangerous collecting strategy to one of the lower Gardens' most specialised grazing relationships.",
    ],
);

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
