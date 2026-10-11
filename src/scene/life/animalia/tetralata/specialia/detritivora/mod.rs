pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod lucentiae;
pub mod manducidae;

pub const TAXON: Taxon = Taxon {
    name: "Detritivora",
    english_name: "Debris Eaters",
    description: "Debris Eaters collect nutrients from suspended waste, organic particles and the bodies of larger animals. \
        Small agile cleaners and swarm decomposers share the order with elongated filter feeders. \
        Their feeding recycles material that would otherwise fall out of the inhabited sky.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
