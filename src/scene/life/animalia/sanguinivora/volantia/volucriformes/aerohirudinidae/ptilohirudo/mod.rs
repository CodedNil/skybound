pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod pennata;

pub const TAXON: Taxon = Taxon {
    name: "Ptilohirudo",
    english_name: "Feathered Leech",
    description: "Feathered Leeches are small gliders whose feathered wings support quick approaches and retreats through cloud layers.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
