pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod taeniatus;

pub const TAXON: Taxon = Taxon {
    name: "Levitans",
    english_name: "Levitator",
    description: "Drifters have a flattened, leaf-like body with a pointed leading edge and long trailing ribbons.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
