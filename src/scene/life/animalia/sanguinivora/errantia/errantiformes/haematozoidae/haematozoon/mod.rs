pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod invasor;

pub const TAXON: Taxon = Taxon {
    name: "Haematozoon",
    english_name: "Blood Animal",
    description: "Blood Dwellers are parasites that invade and develop within a host's bloodstream.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
