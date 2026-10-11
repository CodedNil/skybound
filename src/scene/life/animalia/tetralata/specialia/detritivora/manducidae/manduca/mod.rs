pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod gregaria;

pub const TAXON: Taxon = Taxon {
    name: "Manduca",
    english_name: "Muncher",
    description: "Nibblers are tiny, opportunistic four-winged scavengers that gather in large feeding swarms.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
