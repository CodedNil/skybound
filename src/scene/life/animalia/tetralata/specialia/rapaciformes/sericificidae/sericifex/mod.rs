pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod textor;

pub const TAXON: Taxon = Taxon {
    name: "Sericifex",
    english_name: "Silk Maker",
    description: "Weavers are aerial engineers that spin silk between separate plants, turning loose vegetation into connected habitats.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
