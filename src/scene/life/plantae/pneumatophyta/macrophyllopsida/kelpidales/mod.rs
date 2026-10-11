pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod kelpiaceae;
pub mod laevensailiaceae;

pub const TAXON: Taxon = Taxon {
    name: "Kelpidales",
    english_name: "Kelp-like",
    description: "Kelp-like plants use broad membranes and fibrous structures to occupy extensive areas of the sky. \
        Distributed buoyancy supports their broad surfaces along the length of the plant. \
        The order includes ribbon forests, horizontal sails and concave leaves that retain water.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
