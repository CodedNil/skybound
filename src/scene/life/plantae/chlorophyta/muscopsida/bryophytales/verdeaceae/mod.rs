pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod aeromuscus;

pub const TAXON: Taxon = Taxon {
    name: "Verdeaceae",
    english_name: "Green Cover",
    description: "Green Cover plants carpet suitable surfaces with dense velvety growth. \
        Small leaves collect light and moisture while slow decay returns nutrients to the surrounding mat. \
        Bark and rough shells provide secure attachment and let the plants travel with their hosts.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
