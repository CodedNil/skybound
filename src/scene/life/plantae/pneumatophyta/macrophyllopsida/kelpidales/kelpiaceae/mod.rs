pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod kelpus;

pub const TAXON: Taxon = Taxon {
    name: "Kelpiaceae",
    english_name: "Kelps",
    description: "Kelps grow long fibrous ribbons supported by many regularly spaced gas bladders. \
        Repeated lift points help strands remain broadly level despite their great length. \
        Dense stands form aerial forests that offer food, shelter and a framework for other attached organisms.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
