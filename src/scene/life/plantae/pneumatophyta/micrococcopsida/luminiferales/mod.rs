pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod luminiferaceae;

pub const TAXON: Taxon = Taxon {
    name: "Luminiferales",
    english_name: "Light Bearers",
    description: "Light Bearers form small luminous growths that use ambient aur and live on larger organisms. \
        Fast expansion and fragmentation allow patches to spread across suitable surfaces. \
        Their light and edible tissues connect host-associated growth to the surrounding aerial food web.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
