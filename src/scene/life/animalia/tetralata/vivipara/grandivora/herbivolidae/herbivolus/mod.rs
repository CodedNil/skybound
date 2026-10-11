pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod migratorius;

pub const TAXON: Taxon = Taxon {
    name: "Herbivolus",
    english_name: "Plant Flier",
    description: "Grazers are large, slow-moving herbivores with broad feathered wings and long, soft tongue-like feeding appendages.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
