pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod macrophyllopsida;
pub mod micrococcopsida;

pub const TAXON: Taxon = Taxon {
    name: "Pneumatophyta",
    english_name: "Air Plants",
    description: "Air-Float Plants remain aloft through a combination of low mass, lifting surfaces and internal gas. \
        Their bodies range from individual drifting cells to immense fibrous ribbons and spreading membranes. \
        Updrafts distribute microscopic forms, while larger plants spread buoyancy across multiple chambers to remain stable.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
};
