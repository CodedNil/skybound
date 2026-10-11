pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod auracoccales;
pub mod luminiferales;

pub const TAXON: Taxon = Taxon {
    name: "Micrococcopsida",
    english_name: "Small Spheres",
    description: "Small Spheres are microscopic plants that drift freely or form fine growths on larger organisms. \
        Their small size and low mass allow passive suspension in rising air. \
        Rapid reproduction and efficient energy collection make these forms abundant food and important partners or parasites of larger life.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
