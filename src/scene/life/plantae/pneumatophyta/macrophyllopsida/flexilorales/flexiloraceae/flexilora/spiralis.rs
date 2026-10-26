use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    plantae::{frond, rotate},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, site},
    plantae::physiology::BLADDER,
    plantae::physiology::FROND,
    plantae::physiology::ROOT,
    plantae::physiology::STEM,
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.31,
    maturity_days: 30.0,
    lifespan_days: 4000.0,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 12.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.93, -1.25), vec3(0.06, 0.06, 0.06)),
    site(BLADDER, vec3(0.00, 2.64, -1.39), vec3(0.08, 0.08, 0.08)),
    site(FROND, vec3(1.36, 0.14, 0.36), vec3(0.07, 0.07, 0.07)),
    site(ROOT, vec3(0.00, -4.36, -1.61), vec3(0.10, 0.10, 0.10)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.24, 0.44, 0.18), vec3(0.6, 0.7, 0.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation, epiphyte};

    pub const PHYSIOLOGY: Physiology = Physiology {
        powered: |climate| climate.wind.length_squared() * 0.03,
        propagation: Propagation::Fragmentation,
        attaches: true,
        partner: epiphyte,
        floats: false,
        ..super::super::PHYSIOLOGY
    };
}
