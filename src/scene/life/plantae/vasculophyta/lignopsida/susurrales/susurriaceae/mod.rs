pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod susurrium;

pub const TAXON: Taxon = Taxon {
    name: "Susurriaceae",
    english_name: "Whispers",
    description: "Whispers grow hollow, ridged trunks and tubular branches with delicate fan leaves. \
        Air moving through branch openings produces individual whistles, creating an audible signature for each tree. \
        Sap networks also carry signals to resident Glowgnomes, whose coordinated wing movements help the host reposition.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
