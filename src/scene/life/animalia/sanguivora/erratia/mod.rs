pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod errantia;

pub const TAXON: Taxon = Taxon {
    name: "Erratia",
    english_name: "Wanderers",
    description: "Wanderers spend their unattached lives being carried between potential hosts by wind or floating vegetation. \
        Their soft, elongated bodies favour concealment and secure attachment over powered flight. \
        Once a suitable host is reached, many settle permanently and alter their appearance to match it.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
