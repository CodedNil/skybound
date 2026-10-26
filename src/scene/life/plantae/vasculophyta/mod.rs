pub use super::render;
use crate::scene::life::biology::Biology;
pub mod floropsida;
pub mod lignopsida;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Phylum,
    "Vasculophyta",
    "Vessel Plants",
    [
        "Members of this phylum connect distant tissues through internal vascular channels.",
        "These networks supply flowers, woody supports and hanging roots while exposed surfaces recover water and dissolved minerals from the air.",
        "Buoyant organs, broad leaves and hollow structures provide support without a rooted ground stage.",
        "Reproductive branches exchange pollen and package seeds, while other lineages retain spores or reproduce through living fragments.",
        "Their evolution permits large organised bodies whose specialised regions share resources across distances too great for surface exchange alone.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        vascular: true,
        ..super::super::PHYSIOLOGY
    };
}
