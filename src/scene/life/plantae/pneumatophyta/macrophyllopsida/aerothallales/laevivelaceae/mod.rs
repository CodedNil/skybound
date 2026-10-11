use crate::scene::life::biology::{Biology, Lift, Organs};
pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Membrane,
        ..super::BIOLOGY.organs
    },
    ..super::BIOLOGY
};
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod lacufolium;
pub mod laevivelum;

pub const TAXON: Taxon = Taxon {
    name: "Laevivelaceae",
    english_name: "Smooth Sails",
    description: "Smooth Sails grow broad, thin collecting membranes that float mainly horizontally. \
        Their spread surfaces create shade and cooler air beneath overlapping plants. \
        Some retain the flat sail form, while others curve into protective water-holding basins.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
