use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 30.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 15.0;

pub const TAXON: Taxon = Taxon {
    name: "Lacufolium concavum",
    english_name: "Pondplants",
    description: "Pondplants turn a broad floating leaf into a concave basin that stores rainwater. \
        Their shallow aerial ponds are crucial drinking sites for animals that cannot extract enough moisture themselves. \
        Water stored in the basin also discourages grazing: damage to the leaf spills the contents and disrupts the feeding surface.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

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

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.24, 0.5, 0.17),
        vec3(0.65, 0.79, 0.3),
    )
}
