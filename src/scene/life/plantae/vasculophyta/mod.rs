pub use super::render;
use crate::scene::life::biology::{Biology, Organs, Transport};
pub mod floropsida;
pub mod lignopsida;

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        transport: Transport::Vascular,
        ..super::BIOLOGY.organs
    },
    ..super::BIOLOGY
};

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
