pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod concavum;

pub const TAXON: Taxon = Taxon {
    name: "Lacufolium",
    english_name: "Lake Leaf",
    description: "Pondplants turn a broad floating leaf into a concave basin that stores rainwater.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
