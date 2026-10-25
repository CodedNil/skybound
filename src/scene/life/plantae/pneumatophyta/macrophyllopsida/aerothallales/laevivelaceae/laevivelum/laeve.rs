use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    surface::{Ray, Surface, TraceResult},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 730.0,
    lifespan_days: 60000.0,
    adult_mass: 18.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 17.0;

taxon!(
    Species,
    "Laevivelum laeve",
    "Smooth Sails",
    [
        "Smooth Sails spread broad, thin aur-collecting membranes almost horizontally through rising air.",
        "Gentle undulation changes drag and sheds uneven loads of rain before they distort the whole surface.",
        "Small marginal glands produce a soft glow when the surrounding aur light dims.",
        "Overlapping sails cool and shelter the air below, creating patches useful to moisture-sensitive organisms.",
        "Windborne spores establish new membranes, while browsing Old Wise Ones keep mature sails connected to the wider food web.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(11.0, 3.0, 11.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let radius = p.xz().length();
    let height =
        (radius * 0.65).sin() * 0.25 + (p.x * 0.7 + c.animation.z * 0.4).sin() * radius * 0.035;
    let mut shape = Surface::new(
        (p.y - height).abs().max((radius - 9.0) * 0.5) * 0.65 - 0.045,
        1,
    );
    for i in 0..20 {
        let angle = i as f32 * TAU / 20.0;
        shape.join(
            ellipsoid(
                p - vec3(
                    angle.cos() * 8.9,
                    (8.9f32 * 0.65).sin() * 0.25,
                    angle.sin() * 8.9,
                ),
                Vec3::splat(0.14),
            ),
            2,
        );
    }
    shape
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.17, 0.5, 0.36), vec3(0.29, 0.95, 0.7));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
