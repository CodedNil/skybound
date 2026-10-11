pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod vexillariaceae;

pub const TAXON: Taxon = Taxon {
    name: "Fulminatales",
    english_name: "Lightning Strikers",
    description: "Lightning Strikers harvest electrical energy in storm-prone air. \
        Woody supporting structures carry broad sails that reposition the plant between charging opportunities. \
        Their movements link energy collection to the changing location of storms.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
