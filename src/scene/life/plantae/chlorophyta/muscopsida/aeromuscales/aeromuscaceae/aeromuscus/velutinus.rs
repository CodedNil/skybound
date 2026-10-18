use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 5.0,
    lifespan_days: 1000.0,
    adult_mass: 0.05,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 9.0;

taxon!(
    Species,
    "Aeromuscus velutinus",
    "Air Moss",
    [
        "Air Moss carpets bark and rough animal shells with dense green leaves that hold dew between their tips.",
        "The mat uses scattered aur light and dissolved nutrients retained from dust and decaying debris.",
        "Its low profile keeps most growth within the sheltered air close to a moving host.",
        "Tiny animals find moisture and refuge among the leaves, while slow decomposition returns their waste to the plant.",
        "Detached pieces can start another patch, making close passages between living communities important opportunities for dispersal.",
    ],
);

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
