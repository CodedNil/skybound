pub use super::render;
use crate::scene::life::biology::Biology;
pub mod muscopsida;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Phylum,
    "Chlorophyta",
    "Green Plants",
    [
        "Members of this phylum use pigments to convert scattered aur light into chemical energy.",
        "Thin moisture-retaining tissues exchange gases and dissolved nutrients directly with the surrounding air.",
        "Their attached growth needs no independent lifting body, allowing dense colonies to occupy bark and rough animal coverings.",
        "Windborne fragments establish new patches when they reach a surface that holds moisture without remaining deeply flooded.",
        "Dependence on light distinguishes them from plants that collect dispersed aur directly, making heavy mineral haze a limit even where nutrients are abundant.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation, epiphyte};

    pub const PHYSIOLOGY: Physiology = Physiology {
        powered: |climate| climate.light,
        propagation: Propagation::Fragmentation,
        attaches: true,
        partner: epiphyte,
        floats: false,
        ..super::super::PHYSIOLOGY
    };
}
