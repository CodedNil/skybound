pub use super::{BIOLOGY, render_wings};

pub mod lucens;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Candela",
    "Candle",
    [
        "Members of Candela are luminous sap feeders whose association with woody hosts has become mutualistic.",
        "Their small bodies fit within cavities that protect them from drying, predators and the strongest crosswinds.",
        "Sap carries both their food and the chemical signals that coordinate groups around the host.",
        "A single pair of wings can produce useful collective force when many animals respond together.",
        "Light, controlled feeding and coordinated movement distinguish this lineage from relatives that only disguise the cost of parasitism.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Glowgnomes court with the lamps they carry, hanging together in the dark of the wood.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Situation, display_near},
            nature::ROOST,
            sanguinivora::ETHOLOGY as BLOOD_FEEDER,
        },
        behaviour::Goal,
    };

    pub const ETHOLOGY: Ethology = Ethology {
        court: glow,
        ..BLOOD_FEEDER
    };

    fn glow(s: &Situation) -> Goal {
        display_near(s.seen, 2.0, s.seen.mark(ROOST))
    }
}
