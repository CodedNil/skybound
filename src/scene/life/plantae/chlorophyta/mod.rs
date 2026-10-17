use crate::scene::life::biology::{Biology, Energy, Reproduction};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Attached,
        ..super::BIOLOGY.organs
    },
    energy: Energy::Light,
    reproduction: Reproduction::Fragmentation,
    ..super::BIOLOGY
};

pub use super::render;

pub mod muscopsida;

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
