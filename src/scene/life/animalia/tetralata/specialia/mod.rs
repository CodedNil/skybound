pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod detritivora;
pub mod rapaxina;

pub const TAXON: Taxon = Taxon {
    name: "Specialia",
    english_name: "Particular Ones",
    description: "Particular Ones have specialised into ecological tasks such as decomposition, cleaning, filtering and silk construction. \
        Body plans vary from tiny swarm animals to long glowing drifters and armoured predators, while retaining four-winged ancestry. \
        Their activity recycles waste and helps connect otherwise isolated aerial habitats.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
