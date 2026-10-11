use crate::scene::life::biology::{Biology, Energy, Reproduction};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    energy: Energy::Wind,
    reproduction: Reproduction::Fragmentation,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}
use crate::scene::life::plantae::{frond, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Flexilora",
    english_name: "Cicaia",
    description: "Cicaia forms long spiralling tendrils attached to trees or animals. \
        Piezoelectric fibres convert bending, twisting and stretching in the wind into chemical energy. \
        The flexible coils stretch and twist with the movement of their host and the surrounding currents. \
        Detached fragments can grow into new individuals, allowing ordinary breakage to spread the plant.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 12.0,
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 8.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let y = p.y.clamp(-6.0, 6.0);
    let angle = y * 1.8 + c.animation.z * 0.15;
    let radius = 1.5 + 0.18 * (y * 1.3).sin();
    let center = vec3(angle.cos() * radius, y, angle.sin() * radius);
    let mut shape = Surface::new((p - center).length() * 0.35 - 0.13, 1);
    for i in -4..=4 {
        let angle = i as f32 * 2.1;
        let root = vec3(angle.cos() * 1.5, i as f32 * 1.3, angle.sin() * 1.5);
        shape.join(frond(rotate(p - root, angle), 2.0, 0.4, 0.8, 3.0), 1);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, vec3(0.24, 0.44, 0.18), vec3(0.6, 0.7, 0.2))
}
