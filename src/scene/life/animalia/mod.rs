use spirv_std::glam::{Vec3, Vec4, vec3};

use crate::scene::life::{CreatureInstance, Rank, Taxon};
use crate::scene::surface::{Ray, Surface, TraceResult, paired_sheets, trace};

pub mod bullidorsata;
pub mod sanguivora;
pub mod tetralata;

pub const TAXON: Taxon = Taxon {
    name: "Animalia",
    english_name: "Animals",
    description: "Animals acquire nutrients from other organisms and organic debris, ranging from passive sponge-like drifters to intelligent aerial predators. \
        Their bodies exploit the planet's continuous updraft through gas chambers, lifting surfaces or powered wings, and many provide living habitat for smaller animals and plants.",
    rank: Rank::Kingdom,
    parent: Some(&super::TAXON),
    radius: 0.0,
};

/// Render the body and wing pairs (span, chord, root Z, opacity).
pub fn render_wings<const N: usize>(
    ray: Ray,
    c: &CreatureInstance,
    surface: impl Fn(Vec3, &CreatureInstance) -> Surface,
    pigment: Vec4,
    accent: Vec3,
    wings: [Vec4; N],
) -> TraceResult {
    let mut result = TraceResult::opaque(trace(ray, c, surface, pigment, accent));
    for wing in wings {
        paired_sheets(
            &mut result,
            ray,
            c,
            vec3(0.0, 0.5, wing.z),
            wing.truncate().truncate(),
            accent.extend(wing.w),
        );
    }
    result
}
