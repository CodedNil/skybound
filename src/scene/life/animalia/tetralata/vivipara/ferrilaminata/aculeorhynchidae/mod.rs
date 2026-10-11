pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod aculeorhynchus;

pub const TAXON: Taxon = Taxon {
    name: "Aculeorhynchidae",
    english_name: "Spined Beaks",
    description: "Spined Snouts concentrate their predatory weapon at the front of the head. \
        Long, rigid snout structures align with a streamlined body and powerful diving flight. \
        The head and body align during a steep descent, concentrating the force of the strike at the snout.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
