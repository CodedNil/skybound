pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod tenax;

pub const TAXON: Taxon = Taxon {
    name: "Adhaerens",
    english_name: "Clinger",
    description: "Attached Ones begin as windborne leeches and settle permanently when they encounter an animal.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
