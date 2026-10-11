pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod tempestivagaceae;

pub const TAXON: Taxon = Taxon {
    name: "Tempestivagales",
    english_name: "Storm Wanderers",
    description: "Lightning Strikers harvest electrical energy in storm-prone air. \
        Woody supporting structures carry broad sails that reposition the plant between charging opportunities. \
        Their movements link energy collection to the changing location of storms.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
