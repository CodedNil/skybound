pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod ferrilaminata;
pub mod grandivoriformes;

pub const TAXON: Taxon = Taxon {
    name: "Viviparia",
    english_name: "Live Bearers",
    description: "Live Bearers include some of the sky's largest and most intelligent animals. \
        Many communicate with clicks, whistles or songs, supporting social groups, migration and coordinated responses to danger. \
        Their four-winged bodies range from slow feathered grazers to agile predators and aur-adapted mutualists.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
