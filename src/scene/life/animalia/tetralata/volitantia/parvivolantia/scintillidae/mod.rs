use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    social: Social::Swarm,
    adult_mass: 0.005,
    speed: 11.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod scintilla;

pub const TAXON: Taxon = Taxon {
    name: "Scintillidae",
    english_name: "Sparkies",
    description: "Minute luminous herbivores feed on algae and seeds in coordinated aerial swarms.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
