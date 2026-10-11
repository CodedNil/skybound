pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod pollinifloraceae;

pub const TAXON: Taxon = Taxon {
    name: "Polliniflorales",
    english_name: "Pollen Flowers",
    description: "Nectar-bearing plants store energy in flowers whose colour, light and sweetness attract visiting animals. \
        Petals present broad visual signals, while nectar reservoirs reward contact with pollen-bearing structures. \
        The order ranges from delicate small blooms to flowers as large as trees.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
