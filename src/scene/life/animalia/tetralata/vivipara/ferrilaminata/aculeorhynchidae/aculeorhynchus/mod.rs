pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod acuminatus;

pub const TAXON: Taxon = Taxon {
    name: "Aculeorhynchus",
    english_name: "Spined Beak",
    description: "Needlesharks carry a huge needle-like spike on the head of a streamlined four-winged body.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
