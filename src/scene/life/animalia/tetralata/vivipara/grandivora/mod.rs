pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod herbivolidae;
pub mod vocigenidae;

pub const TAXON: Taxon = Taxon {
    name: "Grandivora",
    english_name: "Large Feeders",
    description: "Large Feeders use extensive wings and large bodies to travel between scattered concentrations of food. \
        Social behaviour, long-distance communication and memory help them exploit predictable migrations and feeding grounds. \
        Broad feathering, dark scavenging plumage or extensive pale fur distinguish the major families.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
