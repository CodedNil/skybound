pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod lucinani;

pub const TAXON: Taxon = Taxon {
    name: "Candelis",
    english_name: "Little Candles",
    description: "Little Candles are luminous descendants of parasites whose association with plants has become mutualistic. \
        Small bodies fit into sheltered cavities, and their light helps their plant hosts attract pollinators. \
        Sap carries both nutrients and signals that coordinate the animals' activity.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
