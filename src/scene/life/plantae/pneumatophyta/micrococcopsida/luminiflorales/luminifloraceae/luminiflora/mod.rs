pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod symbiotica;

pub const TAXON: Taxon = Taxon {
    name: "Luminiflora",
    english_name: "Luminous Flower",
    description: "Lumaweed is a widespread, rapidly growing bioluminescent plant that forms patches on larger lifeforms.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
