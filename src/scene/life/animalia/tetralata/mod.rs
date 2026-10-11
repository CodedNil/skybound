pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod specialia;
pub mod viviparia;
pub mod volitilia;

pub const TAXON: Taxon = Taxon {
    name: "Tetralata",
    english_name: "Four Winged",
    description: "Four Winged animals share two pairs of lifting surfaces, modified into membranes, feathers or heavier specialised structures. \
        The phylum includes luminous swarm animals, decomposers, silk engineers and large intelligent live bearers. \
        Wing proportions, armour and body shape reflect very different ways of exploiting the same rising air.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
