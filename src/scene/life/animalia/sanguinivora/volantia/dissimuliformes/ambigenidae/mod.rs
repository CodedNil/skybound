pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod ambigena;
pub mod candela;

pub const TAXON: Taxon = Taxon {
    name: "Ambigenidae",
    english_name: "Double-natured Ones",
    description: "Mimics resemble aerial cleaners or luminous host companions. \
        Specialised mouth fringes and restrained feeding allow some to remove debris while also siphoning fluids. \
        In other branches the exchange has become beneficial to both partners, linking sap feeding, light production and movement of living plants.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
