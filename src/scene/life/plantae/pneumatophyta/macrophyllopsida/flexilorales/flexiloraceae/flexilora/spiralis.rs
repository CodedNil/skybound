use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::CreatureInstance;
use crate::scene::life::biology::{Biology, Energy, Lift, Organs, Reproduction};
use crate::scene::life::plantae::{frond, rotate};
use crate::scene::life::surface::{Ray, Surface, TraceResult};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 30.0,
    lifespan_days: 4000.0,
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    energy: Energy::Wind,
    reproduction: Reproduction::Fragmentation,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 12.0;

taxon!(
    Species,
    "Flexilora spiralis",
    "Cicaia",
    [
        "Cicaia forms long spiralling tendrils that attach to trees or animals while loose coils move in the passing air.",
        "Piezoelectric fibres convert stretching, bending and twisting into chemical energy.",
        "Several flexible contact points spread the load, letting the plant exploit a host's motion without making every gust a tearing force.",
        "Strongly sheared currents near Rivers provide abundant movement but also increase the risk of breakage.",
        "Surviving fragments can attach elsewhere and grow, turning that mechanical risk into a means of reproduction.",
    ],
);

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
