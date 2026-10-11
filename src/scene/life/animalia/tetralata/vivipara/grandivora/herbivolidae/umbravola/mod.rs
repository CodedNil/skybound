pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod tacita;

pub const TAXON: Taxon = Taxon {
    name: "Umbravola",
    english_name: "Shadow Flier",
    description: "Shadowings are large, intelligent scavengers covered in muted, light-absorbing dark plumage.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
