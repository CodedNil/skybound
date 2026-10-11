pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod nebulaphoridae;
pub mod scintillidae;

pub const TAXON: Taxon = Taxon {
    name: "Parvivolantia",
    english_name: "Small Flyer",
    description: "Small Flyers are lightweight four-winged animals associated with suspended food and flowering plants. \
        Their wings provide manoeuvrability around feeding sites, while luminous or translucent surfaces are prominent visual traits. \
        Swarms and long-distance pollen transport make them important links between scattered plant communities.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
