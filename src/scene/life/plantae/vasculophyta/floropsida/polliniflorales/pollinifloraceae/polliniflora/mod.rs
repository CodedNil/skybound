pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod nectarifera;

pub const TAXON: Taxon = Taxon {
    name: "Polliniflora",
    english_name: "Pollen Flower",
    description: "Blooms are delicate flowering plants with layered petals and reservoirs of sweet, aur-rich nectar.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
