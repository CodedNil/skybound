pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod mutualis;

pub const TAXON: Taxon = Taxon {
    name: "Purgator",
    english_name: "Cleaner",
    description: "Sweepers are small, bloated animals that flit around larger creatures using specialised mouth palps to remove parasites and debris.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
