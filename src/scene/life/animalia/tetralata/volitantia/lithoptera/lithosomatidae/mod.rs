use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 1.0,
    adult_mass: 80.0,
    altitude: (60.0, 1500.0),
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod lithosoma;

pub const TAXON: Taxon = Taxon {
    name: "Lithosomatidae",
    english_name: "Stone Bodies",
    description: "Dense mineral-crusted herbivores drift through lower currents, feeding on root-stars and mineral-rich growth.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
