use crate::scene::life::biology::{Biology, Organs, Transport};
pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        transport: Transport::Vascular,
        ..super::BIOLOGY.organs
    },
    ..super::BIOLOGY
};
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod floropsida;
pub mod lignopsida;

pub const TAXON: Taxon = Taxon {
    name: "Vasculophyta",
    english_name: "Vessel Plants",
    description: "Vessel Bearers have internal vascular tissues that move fluids and support larger, more organised bodies. \
        These networks supply woody stems, hanging roots, flowers and fruit while keeping distant tissues connected. \
        Hollow structures, buoyant organs and broad leaves support their aerial bodies.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
};
