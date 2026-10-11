pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod velutinus;

pub const TAXON: Taxon = Taxon {
    name: "Aeromuscus",
    english_name: "Air Moss",
    description: "Air Moss forms dense green velvet-like mats on bark and occasionally on rough-shelled animals.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
