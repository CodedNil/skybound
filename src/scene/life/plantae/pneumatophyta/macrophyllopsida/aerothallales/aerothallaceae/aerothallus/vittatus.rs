use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 365.0,
    lifespan_days: 40000.0,
    adult_mass: 25.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{frond, rotate};
use crate::scene::life::{CreatureInstance, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 25.0;

taxon!(
    Species,
    "Aerothallus vittatus",
    "Kelp",
    [
        "Kelp spreads long fibrous strands through the Gardens, with gas bladders supporting broad aur-collecting fronds at repeated intervals.",
        "Each strand bends across several neighbouring currents, distributing lift instead of hanging from a single buoyant crown.",
        "Dense growth interweaves into Jungles whose sheltered interior holds animals, attached plants and Weaver corridors.",
        "Grazers and Old Wise Ones browse the outer fronds, while broken tissue feeds smaller scavengers downstream.",
        "Spores establish new floating strands, allowing a forest to spread without a trunk rooted to any permanent surface.",
    ],
);

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
