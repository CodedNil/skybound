pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod erratia;
pub mod voluceria;

pub const TAXON: Taxon = Taxon {
    name: "Sanguivora",
    english_name: "Blood Eaters",
    description: "Blood Eaters descend from soft-bodied, sucker-bearing parasites that obtain fluids from living hosts. \
        Their elongated bodies range from camouflaged, permanently attached leeches to agile winged visitors. \
        Some lineages disguise feeding as grooming; others have evolved mutualistic relationships with their hosts.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
