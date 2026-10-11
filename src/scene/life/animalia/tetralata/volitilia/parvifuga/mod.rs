pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod luminiscidae;
pub mod nebulaphoridae;

pub const TAXON: Taxon = Taxon {
    name: "Parvifuga",
    english_name: "Small Flyer",
    description: "Small Flyers are lightweight four-winged animals associated with suspended food and flowering plants. \
        Their wings provide manoeuvrability around feeding sites, while luminous or translucent surfaces are prominent visual traits. \
        Swarms and long-distance pollen transport make them important links between scattered plant communities.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
