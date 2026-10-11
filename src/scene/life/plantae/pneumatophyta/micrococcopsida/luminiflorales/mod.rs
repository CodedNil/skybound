pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod luminifloraceae;

pub const TAXON: Taxon = Taxon {
    name: "Luminiflorales",
    english_name: "Luminous Flowers",
    description: "Light Bearers form small luminous growths that use ambient aur and live on larger organisms. \
        Fast expansion and fragmentation allow patches to spread across suitable surfaces. \
        Their light and edible tissues connect host-associated growth to the surrounding aerial food web.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
