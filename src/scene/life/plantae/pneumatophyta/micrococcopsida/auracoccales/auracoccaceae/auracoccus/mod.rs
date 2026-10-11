pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod communis;

pub const TAXON: Taxon = Taxon {
    name: "Auracoccus",
    english_name: "Aur Sphere",
    description: "Algae are diverse microscopic plants suspended as individual particles in the updraft.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
