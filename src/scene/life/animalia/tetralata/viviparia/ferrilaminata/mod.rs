use crate::scene::life::biology::{Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Predator,
    speed: 45.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod aculeorhinidae;
pub mod ferrivolvidae;
pub mod naeturidae;

pub const TAXON: Taxon = Taxon {
    name: "Ferrilaminata",
    english_name: "Iron Blades",
    description: "Iron Blades includes agile four-winged hunters and their aur-adapted relatives. \
        Pointed feeding structures, cutting blades or specialised protective tissues distinguish its families. \
        Sharp banking and diving are common flight behaviours; the Næturidae also possess specialised aur-processing tissues and close social bonds.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
