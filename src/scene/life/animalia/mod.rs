use spirv_std::glam::{Vec3, Vec4, vec3};

use crate::scene::life::CreatureInstance;
use crate::scene::life::biology::{Anatomy, Biology};
use crate::scene::life::surface::{Ray, Surface, TraceResult, paired_sheets, trace};
pub mod bullidorsata;
pub mod sanguinivora;
pub mod tetralata;

pub const BIOLOGY: Biology = Biology {
    anatomy: Anatomy::Porous,
    ..super::BIOLOGY
};

taxon!(
    Kingdom,
    "Animalia",
    "Animals",
    [
        "Animals obtain usable energy by consuming organic material made by other organisms.",
        "Porous forms absorb food across wet surfaces, while active lineages gather it into a digestive crop and distribute nutrients through pulsatile channels.",
        "Moisture-recovery folds conserve water, although sustained flight can still force animals to seek Pondplants or humid currents.",
        "Lift comes from gas chambers, wings or attachment to another living body, and every lineage must survive the weaker updraft of the dim aur phase.",
        "Direct exposure to concentrated aur remains lethal except within the carefully limited mist-skimming adaptations of the Nætu lineage.",
    ],
);

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
