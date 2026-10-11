pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod grave;

pub const TAXON: Taxon = Taxon {
    name: "Lithosoma",
    english_name: "Stone Body",
    description: "Grounders are massive, mineral-crusted herbivores with rocky hides and broad lifting surfaces.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
