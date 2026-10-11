pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod scintillans;

pub const TAXON: Taxon = Taxon {
    name: "Scintilla",
    english_name: "Spark",
    description: "Sparkies are tiny, intensely luminous four-winged animals that gather into vast shimmering swarms.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
