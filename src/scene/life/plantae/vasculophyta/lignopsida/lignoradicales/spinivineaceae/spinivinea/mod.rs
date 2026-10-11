pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod volitans;

pub const TAXON: Taxon = Taxon {
    name: "Spinivinea",
    english_name: "Thorn Vine",
    description: "Brambles form dense spheres or chains of thorny woody vines that drift independently on strong currents.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
