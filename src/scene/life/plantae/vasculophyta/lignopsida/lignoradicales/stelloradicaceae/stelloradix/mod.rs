pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod pendula;

pub const TAXON: Taxon = Taxon {
    name: "Stelloradix",
    english_name: "Star Root",
    description: "Root-Stars have a buoyant rubbery sphere above a star-shaped cluster of thick fleshy roots.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
