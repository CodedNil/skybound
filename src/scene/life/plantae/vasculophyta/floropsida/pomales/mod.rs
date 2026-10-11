pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod pomaceae;

pub const TAXON: Taxon = Taxon {
    name: "Pomales",
    english_name: "Fruit-bearing",
    description: "Fruit-bearing plants package stored nutrients and seeds into buoyant bodies. \
        Thick collecting rinds support growth while allowing fruit fields to drift together through suitable air. \
        Ripening releases seeds gently, and specialised seed coverings use currents and passing animals to travel farther.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
