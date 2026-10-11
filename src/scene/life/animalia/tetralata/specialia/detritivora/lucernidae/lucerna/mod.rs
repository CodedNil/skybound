pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod undulata;

pub const TAXON: Taxon = Taxon {
    name: "Lucerna",
    english_name: "Lamp",
    description: "Shimmerays are long, softly bioluminescent drifters with broad wings and trailing structures.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
