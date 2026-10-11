pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod fulgurivora;

pub const TAXON: Taxon = Taxon {
    name: "Tempestivaga",
    english_name: "Storm Wanderer",
    description: "Stormchasers are sail-bearing woody plants common in electrically active regions.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
