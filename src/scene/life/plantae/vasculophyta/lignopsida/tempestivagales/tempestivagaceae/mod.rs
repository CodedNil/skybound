pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod tempestivaga;

pub const TAXON: Taxon = Taxon {
    name: "Tempestivagaceae",
    english_name: "Storm Wanderers",
    description: "Sail-bearers use large membranes to travel through storm-prone regions and present collecting structures to electrical discharges. \
        Slender, sturdy stems support their exposed sails. \
        Sensitivity to approaching storms guides movement toward a valuable but dangerous energy source.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
