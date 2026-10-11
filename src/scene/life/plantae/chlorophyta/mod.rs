pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod muscopsida;

pub const TAXON: Taxon = Taxon {
    name: "Chlorophyta",
    english_name: "Photosynthesisers",
    description: "Photosynthesisers obtain energy primarily from light. \
        Small attached growths collect light across the exposed surfaces of their hosts. \
        Their mats retain moisture and turn weathered bark or rough animal shells into sheltered living microhabitats.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
