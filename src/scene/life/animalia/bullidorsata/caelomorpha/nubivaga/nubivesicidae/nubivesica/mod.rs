pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod caerulea;

pub const TAXON: Taxon = Taxon {
    name: "Nubivesica",
    english_name: "Cloud Bladder",
    description: "Cloud Balloons have a rounded, lobed body dominated by large, closely packed gas chambers.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
