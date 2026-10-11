use crate::scene::life::biology::Biology;
use crate::scene::life::biology::Organs;

pub const BIOLOGY: Biology = Biology {
    organs: Organs::ANIMAL,
    wing_pairs: 1,
    speed: 14.0,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod dissimuliformes;
pub mod volucriformes;

pub const TAXON: Taxon = Taxon {
    name: "Volantia",
    english_name: "Flying Ones",
    description: "Flying Ones retain the flexible bodies and feeding grips of leeches but have evolved wings for repeated journeys between hosts. \
        Wing surfaces range from broad translucent membranes to small feathered gliding structures. \
        Flight permits brief feeding visits and also supports disguises as harmless grooming animals.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
