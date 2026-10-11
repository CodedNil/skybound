pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod gigantea;

pub const TAXON: Taxon = Taxon {
    name: "Gigantiflora",
    english_name: "Giant Flower",
    description: "Gigantaflora produces towering flowers that can approach small trees in height.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
