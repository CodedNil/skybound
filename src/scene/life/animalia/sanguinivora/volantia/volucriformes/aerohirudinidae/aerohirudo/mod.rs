pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod membranacea;

pub const TAXON: Taxon = Taxon {
    name: "Aerohirudo",
    english_name: "Air Leech",
    description: "Air Leeches carry large, translucent, bat-like membrane wings around a narrow flexible body.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
