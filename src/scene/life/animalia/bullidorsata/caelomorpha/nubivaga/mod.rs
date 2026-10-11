pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod levitantidae;
pub mod nubivesicidae;

pub const TAXON: Taxon = Taxon {
    name: "Nubivaga",
    english_name: "Cloud Wanderers",
    description: "Cloud Drifters are light, translucent bubblebacks adapted to open air and cloud margins. \
        Their pale green bodies enclose prominent rounded gas chambers, with gelatinous feeding tissues gathered around the lower centre. \
        Rounded envelopes and trailing membranes catch the light and move gently with surrounding currents. \
        They feed on intercepted plant matter and make small changes in orientation through buoyancy and drag.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
