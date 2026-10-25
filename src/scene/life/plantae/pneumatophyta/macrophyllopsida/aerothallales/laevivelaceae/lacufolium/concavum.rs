use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, Vec3Swizzles, vec3};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1200.0,
    lifespan_days: 90000.0,
    adult_mass: 30.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 15.0;

taxon!(
    Species,
    "Lacufolium concavum",
    "Pondplants",
    [
        "Pondplants curve a broad floating leaf into a shallow basin supported by a flexible outer rim.",
        "Rain accumulates above the collecting membrane, providing drinking water for animals whose moisture-recovery organs cannot meet the cost of active flight.",
        "The basin flexes as its load changes, spilling excess water before the plant loses safe lift.",
        "Hygroscopic tissues supply the plant itself, while the pond remains an exposed resource visited by drinkers, cleaners and opportunistic parasites.",
        "Grazing damage drains this shared water store, so frequent visitors benefit from keeping the living rim intact.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(9.0, 4.0, 9.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let radius = p.xz().length();
    let height = -1.8 + (radius / 7.0).powi(2) * 2.4;
    let mut shape = Surface::new(
        ((p.y - height).abs() - 0.12).max((radius - 7.5) * 0.5) * 0.65,
        1,
    );
    shape.join(
        (p.y + 0.15).abs().max(radius - 5.7).max(-p.y - 1.6) * 0.6 - 0.025,
        4,
    );
    shape.join(ellipsoid(p - vec3(0.0, -1.9, 0.0), vec3(2.2, 0.5, 2.2)), 3);
    shape
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.24, 0.5, 0.17), vec3(0.65, 0.79, 0.3));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
