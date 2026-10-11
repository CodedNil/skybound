pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod laevensails;
pub mod pondphilus;

pub const TAXON: Taxon = Taxon {
    name: "Laevensailiaceae",
    english_name: "Smooth Sails",
    description: "Smooth Sails grow broad, thin collecting membranes that float mainly horizontally. \
        Their spread surfaces create shade and cooler air beneath overlapping plants. \
        Some retain the flat sail form, while others curve into protective water-holding basins.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
