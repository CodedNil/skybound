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
use crate::scene::life::{Rank, Taxon};

pub mod muscopsida;

pub const TAXON: Taxon = Taxon {
    name: "Chlorophyta",
    english_name: "Photosynthesisers",
    description: "Chlorophyta use pigments that convert scattered aur light into chemical energy. \
        Moisture-retaining surfaces shelter the active tissues, which exchange dissolved gases and nutrients with the surrounding air. \
        Their colonies occupy floating bark and animal surfaces, spreading through windborne fragments.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
