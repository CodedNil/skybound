pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod fallax;

pub const TAXON: Taxon = Taxon {
    name: "Ambigena",
    english_name: "Double-natured One",
    description: "Double Natured animals imitate the swollen bodies and specialised mouth palps of Sweepers.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
