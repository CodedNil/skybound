use crate::scene::life::biology::Biology;
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    adult_mass: 0.2,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 10.0;

pub const TAXON: Taxon = Taxon {
    name: "Chromacorallum versicolor",
    english_name: "Color Coral",
    description: "Colour Coral forms bright branching growths on kelp and Laevensail trunks and branches. \
        Frilly fan-like appendages spread greens, yellows, pinks, blues and purples over the host surface. \
        The fans collect aur and emit a faint bioluminescent glow at night. \
        The host provides a stable support and carries the colony through the surrounding air.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

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

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.42, 0.11, 0.55),
        vec3(0.1, 0.8, 0.95),
    )
}
