use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 900.0,
    lifespan_days: 30000.0,
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    adult_mass: 1.0,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 13.0;

taxon!(
    Species,
    "Aurocorallum aureum",
    "Golden Coral",
    [
        "Golden Coral grows layered golden fans from branching attachments on large, slow-moving animals.",
        "Frilled surfaces collect dispersed aur while the host carries the colony through fresh air.",
        "The fans yield in a changing current, reducing strain on their attachments and avoiding the weight of a separate buoyant body.",
        "Moist recesses between branches retain nutrient-bearing dust and shelter small scavengers.",
        "The association remains most stable where modest growth leaves the host's breathing folds and lifting surfaces clear.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, -2.0, 0.0), vec3(1.8, 0.6, 1.8)), 3);
    for i in 0..9 {
        let angle = i as f32 * 2.4;
        let start = vec3(angle.cos(), -2.0, angle.sin());
        let end = vec3(
            angle.cos() * 3.5,
            1.5 + (angle * 2.0).sin(),
            angle.sin() * 3.5,
        );
        shape.join(capsule(p, start, end, 0.22), 3);
        for j in 0..3 {
            let tip = end
                + vec3(
                    (angle + j as f32).cos() * 1.2,
                    j as f32 * 0.7,
                    (angle + j as f32).sin() * 1.2,
                );
            shape.join(capsule(p, end, tip, 0.12), 3);
            shape.join(rosette(rotate(p - tip, angle), 4, 1.0, 0.4, 0.5), 1);
        }
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, vec3(0.75, 0.42, 0.07), vec3(1.0, 0.7, 0.2))
}
