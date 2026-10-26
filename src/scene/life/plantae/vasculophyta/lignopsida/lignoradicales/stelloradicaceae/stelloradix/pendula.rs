use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::rotate,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.7,
    maturity_days: 180.0,
    lifespan_days: 5000.0,
    adult_mass: 7.0,
    altitude: (50.0, 800.0),
};

pub const RENDER_RADIUS: f32 = 16.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.08, -0.06), vec3(0.73, 0.73, 0.73)),
    site(BLADDER, vec3(0.00, -0.49, 1.44), vec3(0.66, 0.66, 0.66)),
    pair(FROND, vec3(1.37, -0.20, -0.37), vec3(0.74, 0.74, 0.74)),
    site(ROOT, vec3(0.00, -0.88, -1.44), vec3(0.49, 0.49, 0.49)),
    site(ROOT, vec3(0.00, 0.49, -1.44), vec3(0.66, 0.66, 0.66)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.48, 0.27, 0.17), vec3(0.7, 0.54, 0.3));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation};

    pub const PHYSIOLOGY: Physiology = Physiology {
        propagation: Propagation::Fragmentation,
        ..super::super::PHYSIOLOGY
    };
}
