pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod floropsida;
pub mod lignopsida;

pub const TAXON: Taxon = Taxon {
    name: "Vasculophyta",
    english_name: "Vessel Bearers",
    description: "Vessel Bearers have internal vascular tissues that move fluids and support larger, more organised bodies. \
        These networks supply woody stems, hanging roots, flowers and fruit while keeping distant tissues connected. \
        Hollow structures, buoyant organs and broad leaves support their aerial bodies.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
