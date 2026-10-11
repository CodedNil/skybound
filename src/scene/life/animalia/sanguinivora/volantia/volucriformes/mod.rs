pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod aerohirudinidae;

pub const TAXON: Taxon = Taxon {
    name: "Volucriformes",
    english_name: "Flyers",
    description: "These winged leeches move between hosts during repeated short feeding visits. \
        Their lightweight bodies and broad lifting surfaces support hovering, gliding and rapid withdrawal. \
        Feeding grounds and cloud layers provide both cover and a supply of potential hosts.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
