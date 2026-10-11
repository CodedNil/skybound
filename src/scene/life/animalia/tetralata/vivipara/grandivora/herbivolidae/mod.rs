pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod herbivolus;
pub mod umbravola;

pub const TAXON: Taxon = Taxon {
    name: "Herbivolidae",
    english_name: "Plant Fliers",
    description: "These large aerial feeders combine substantial bodies with broad feathered appendages and sensitive responses to air currents. \
        Herbivorous branches gather plant material with soft feeding structures, while scavenging relatives search widely for carcasses. \
        Communication and repeated journeys link feeding areas that are far apart in the sky.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
