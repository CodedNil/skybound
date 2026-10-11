use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Predator,
    social: Social::Weaver,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod textricinae;

pub const TAXON: Taxon = Taxon {
    name: "Rapaxina",
    english_name: "Grasping Ones",
    description: "Grasping Ones use trapping structures, specialised mouth palps or projected secretions to capture aerial prey. \
        Their four-winged bodies support precise movement around plants and constructed hunting sites. \
        Trap construction, ambush and precise handling of prey are prominent hunting strategies.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
