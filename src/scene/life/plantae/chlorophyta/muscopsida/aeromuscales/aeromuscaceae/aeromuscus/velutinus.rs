use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 5.0,
    lifespan_days: 1000.0,
    adult_mass: 0.05,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 9.0;

pub const TAXON: Taxon = Taxon {
    name: "Aeromuscus velutinus",
    english_name: "Air Moss",
    description: "Air Moss forms dense green velvet-like mats on bark and occasionally on rough-shelled animals. \
        Fine clustered leaves trap dew and keep the air immediately above the surface moist. \
        Slow decomposition recycles nutrients within the mat. \
        These sheltered patches stabilise microclimates and support tiny insects and other animals that would struggle on an exposed host surface.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 3.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -0.4, 0.0), vec3(3.5, 0.5, 3.5)), 1);
    for i in 0..23 {
        let angle = i as f32 * 2.39996;
        let radius = (i as f32 / 23.0).sqrt() * 3.2;
        let root = vec3(angle.cos() * radius, 0.0, angle.sin() * radius);
        let local = rotate(p - root, angle);
        shape.join(rosette(local, 5, 0.85, 0.16, 0.5), 1);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.12, 0.36, 0.07),
        vec3(0.45, 0.65, 0.15),
    )
}
