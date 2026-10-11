pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod aeromuscus;

pub const TAXON: Taxon = Taxon {
    name: "Aeromuscaceae",
    english_name: "Air Mosses",
    description: "Green Cover plants carpet suitable surfaces with dense velvety growth. \
        Small leaves collect light and moisture while slow decay returns nutrients to the surrounding mat. \
        Bark and rough shells provide secure attachment and let the plants travel with their hosts.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
