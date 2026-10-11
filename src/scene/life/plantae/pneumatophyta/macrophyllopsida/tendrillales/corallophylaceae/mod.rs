pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod aurocoralla;
pub mod chromacorallum;

pub const TAXON: Taxon = Taxon {
    name: "Corallophylaceae",
    english_name: "Coral Leaved",
    description: "Coral Leaved plants build branching attached colonies with broad, frilly collecting fans. \
        Their colourful tissues gather ambient aur while growing over the surfaces of larger organisms. \
        Different branches of the family favour plant trunks or slow-moving animal hosts, producing layered living ornament across the sky.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
