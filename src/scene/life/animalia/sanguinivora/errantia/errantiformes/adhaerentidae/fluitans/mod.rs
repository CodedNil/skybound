pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod crypticus;

pub const TAXON: Taxon = Taxon {
    name: "Fluitans",
    english_name: "Floating One",
    description: "Floating Ones are born among aerial kelp mats, with green-brown colours and thin lateral folds that resemble surrounding vegetation.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
