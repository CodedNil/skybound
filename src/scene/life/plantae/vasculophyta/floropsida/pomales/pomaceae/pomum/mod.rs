pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod iridescens;

pub const TAXON: Taxon = Taxon {
    name: "Pomum",
    english_name: "Fruit",
    description: "Fruit plants drift in fields of large buoyant bodies enclosed by thick iridescent aur-collecting rinds.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
