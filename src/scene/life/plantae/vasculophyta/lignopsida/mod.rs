pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod fulminatales;
pub mod lignoradicales;
pub mod susurrales;

pub const TAXON: Taxon = Taxon {
    name: "Lignopsida",
    english_name: "Wood-bearing",
    description: "Wood-bearing plants develop secondary thickening that produces protective and supporting woody tissues. \
        This permits durable thorny frameworks, tall hollow trunks and structures exposed to storms. \
        Lift still comes from air currents and specialised organs, so strength must be balanced against the cost of weight.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
