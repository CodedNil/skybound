use crate::scene::life::biology::{Biology, Reproduction, Social};

pub const BIOLOGY: Biology = Biology {
    oral_pairs: 3,
    reproduction: Reproduction::LiveBirth,
    social: Social::Herd,
    adult_mass: 12.0,
    perception: 700.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod ferrilaminata;
pub mod grandivora;

pub const TAXON: Taxon = Taxon {
    name: "Vivipara",
    english_name: "Live Bearers",
    description: "Viviparia bear live young within a protected internal brood chamber. \
        Three paired oral tentacles surround the mouth and throat, gathering vegetation or holding prey during feeding. \
        Expanded respiratory folds, water-recovery chambers and pulsatile nutrient sinuses support sustained flight and larger bodies. \
        Gas-storage compartments remain separate from the breathing channels. \
        Social recognition, vocal communication and prolonged care protect slowly developing young.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
