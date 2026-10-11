pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod ambigena;
pub mod candelis;

pub const TAXON: Taxon = Taxon {
    name: "Mimicidae",
    english_name: "Mimics",
    description: "Mimics resemble aerial cleaners or luminous host companions. \
        Specialised limbs and restrained feeding allow some to remove debris while also siphoning fluids. \
        In other branches the exchange has become beneficial to both partners, linking sap feeding, light production and movement of living plants.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
