use crate::scene::life::biology::{Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Nectar,
    adult_mass: 0.1,
    speed: 5.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod nebulaphora;

pub const TAXON: Taxon = Taxon {
    name: "Nebulaphoridae",
    english_name: "Mistings",
    description: "Large docile pollinators travel between floating flowers on broad translucent membrane wings.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
