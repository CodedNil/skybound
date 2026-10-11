pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod aureum;

pub const TAXON: Taxon = Taxon {
    name: "Aurocorallum",
    english_name: "Golden Coral",
    description: "Golden Coral grows as large gold-hued branching colonies with layered, frilled collecting surfaces.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
