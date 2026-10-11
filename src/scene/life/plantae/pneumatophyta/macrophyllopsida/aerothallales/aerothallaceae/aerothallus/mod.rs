pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod vittatus;

pub const TAXON: Taxon = Taxon {
    name: "Aerothallus",
    english_name: "Air Frond",
    description: "Kelp grows as long ribbons of fibrous plant material with small gas bladders spaced along its length.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
