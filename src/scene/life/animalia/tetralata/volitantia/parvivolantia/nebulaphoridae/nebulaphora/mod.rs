pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod vitrea;

pub const TAXON: Taxon = Taxon {
    name: "Nebulaphora",
    english_name: "Mist Bearer",
    description: "Mistings are large, docile moth-like pollinators with two pairs of thin, transparent wings.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
