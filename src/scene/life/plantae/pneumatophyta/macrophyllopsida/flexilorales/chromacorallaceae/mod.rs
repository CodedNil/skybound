pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod aurocorallum;
pub mod chromacorallum;

pub const TAXON: Taxon = Taxon {
    name: "Chromacorallaceae",
    english_name: "Coloured Corals",
    description: "Coral Leaved plants build branching attached colonies with broad, frilly collecting fans. \
        Their colourful tissues gather ambient aur while growing over the surfaces of larger organisms. \
        Different branches of the family favour plant trunks or slow-moving animal hosts, producing layered living ornament across the sky.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
