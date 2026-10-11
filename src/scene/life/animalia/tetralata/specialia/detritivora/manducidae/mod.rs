use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Scavenger,
    social: Social::Swarm,
    speed: 9.0,
    adult_mass: 0.02,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod manduca;
pub mod purgator;

pub const TAXON: Taxon = Taxon {
    name: "Manducidae",
    english_name: "Nibblers",
    description: "Small swarm scavengers recycle suspended organic matter; related cleaners forage on the bodies of larger animals.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
