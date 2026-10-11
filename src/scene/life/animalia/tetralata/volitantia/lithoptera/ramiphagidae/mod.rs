use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 3.0,
    adult_mass: 2.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod ramiphaga;

pub const TAXON: Taxon = Taxon {
    name: "Ramiphagidae",
    english_name: "Branch Eaters",
    description: "Armoured herbivores strip tough woody plant tissue with reinforced feeding structures.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
