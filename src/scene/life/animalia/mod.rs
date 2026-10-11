use crate::scene::life::biology::{Anatomy, Biology};

pub const BIOLOGY: Biology = Biology {
    anatomy: Anatomy::Porous,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, Vec4, vec3};

use crate::scene::life::{CreatureInstance, Rank, Taxon};
use crate::scene::surface::{Ray, Surface, TraceResult, paired_sheets, trace};

pub mod bullidorsata;
pub mod sanguinivora;
pub mod tetralata;

pub const TAXON: Taxon = Taxon {
    name: "Animalia",
    english_name: "Animals",
    description: "Animals obtain chemical energy by digesting organic tissue. \
        Gas-exchange surfaces and moisture-recovery membranes supply metabolism while conserving water; digestive residues carry unassimilated aur back into the currents. \
        Porous lineages distribute nutrients locally, while larger active animals use pulsatile transport channels.",
    rank: Rank::Kingdom,
    parent: Some(&super::TAXON),
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
