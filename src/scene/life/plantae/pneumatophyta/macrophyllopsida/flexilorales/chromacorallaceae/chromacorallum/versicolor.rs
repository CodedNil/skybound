use crate::scene::life::{
    CreatureInstance, Look,
    biology::{Biology, Lift, Organs},
    capsule, ellipsoid, plantae,
    plantae::{rosette, rotate},
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 180.0,
    lifespan_days: 9000.0,
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    adult_mass: 0.2,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 10.0;

taxon!(
    Species,
    "Chromacorallum versicolor",
    "Color Coral",
    [
        "Colour Coral spreads branching fans of green, yellow, pink, blue and purple across kelp and Laevensail growth.",
        "Frilled surfaces collect dispersed aur while narrow attachments leave much of the host exposed to passing air.",
        "Flexible junctions let each fan turn independently when a long frond twists in the current.",
        "A faint glow reveals living margins during dim aur phases and attracts small organisms into the sheltered branches.",
        "The colony gains support and transport, while excessive growth can shade or weigh down the plant carrying it.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 6.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -2.0, 0.0), vec3(0.0, 1.0, 0.0), 0.32),
        0,
    );
    for i in 0..7 {
        let angle = i as f32 * 2.4;
        let root = vec3(0.0, -1.2 + i as f32 * 0.45, 0.0);
        let tip = root + vec3(angle.cos() * 2.3, 1.8, angle.sin() * 2.3);
        shape.join(capsule(p, root, tip, 0.14), 3);
        let local = rotate(p - tip, angle);
        shape.join(rosette(local, 5, 2.1, 0.8, 0.5), 1);
        shape.join(ellipsoid(p - tip, Vec3::splat(0.18)), 2);
    }
    shape
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.42, 0.11, 0.55), vec3(0.1, 0.8, 0.95));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
