pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod laeve;

pub const TAXON: Taxon = Taxon {
    name: "Laevivelum",
    english_name: "Smooth Sail",
    description: "Smooth Sails are flat, broad, thin aurosynthetic membranes that can reach enormous sizes.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
