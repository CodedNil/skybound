pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod dissimuliformes;
pub mod voluceriformes;

pub const TAXON: Taxon = Taxon {
    name: "Voluceria",
    english_name: "Flying Ones",
    description: "Flying Ones retain the flexible bodies and feeding grips of leeches but have evolved wings for repeated journeys between hosts. \
        Wing surfaces range from broad translucent membranes to small feathered gliding structures. \
        Flight permits brief feeding visits and also supports disguises as harmless grooming animals.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
