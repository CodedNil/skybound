pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod voluceridae;

pub const TAXON: Taxon = Taxon {
    name: "Voluceriformes",
    english_name: "Flyers",
    description: "These winged leeches move between hosts during repeated short feeding visits. \
        Their lightweight bodies and broad lifting surfaces support hovering, gliding and rapid withdrawal. \
        Feeding grounds and cloud layers provide both cover and a supply of potential hosts.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
