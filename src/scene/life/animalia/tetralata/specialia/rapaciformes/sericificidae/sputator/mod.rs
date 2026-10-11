pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod acidus;

pub const TAXON: Taxon = Taxon {
    name: "Sputator",
    english_name: "Spitter",
    description: "Spitfire Beetles are small, armoured predators with specialised mouths that launch sticky acidic saliva.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
