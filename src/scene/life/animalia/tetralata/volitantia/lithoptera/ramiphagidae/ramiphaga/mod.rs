pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod corticivora;

pub const TAXON: Taxon = Taxon {
    name: "Ramiphaga",
    english_name: "Branch Eater",
    description: "Driftmunchers are slow, armoured herbivores with strong crushing mandibles and compact supporting wings.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
