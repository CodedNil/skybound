pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod bloomaceae;

pub const TAXON: Taxon = Taxon {
    name: "Nectarales",
    english_name: "Nectar-bearing",
    description: "Nectar-bearing plants store energy in flowers whose colour, light and sweetness attract visiting animals. \
        Petals present broad visual signals, while nectar reservoirs reward contact with pollen-bearing structures. \
        The order ranges from delicate small blooms to flowers as large as trees.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
