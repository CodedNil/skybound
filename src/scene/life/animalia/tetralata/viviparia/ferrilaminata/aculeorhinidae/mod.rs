pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod aculeorhynchus;

pub const TAXON: Taxon = Taxon {
    name: "Aculeorhinidae",
    english_name: "Spined Snout",
    description: "Spined Snouts concentrate their predatory weapon at the front of the head. \
        Long, rigid snout structures align with a streamlined body and powerful diving flight. \
        The head and body align during a steep descent, concentrating the force of the strike at the snout.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
