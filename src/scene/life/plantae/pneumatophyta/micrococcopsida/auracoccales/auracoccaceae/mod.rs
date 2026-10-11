pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod auracoccus;

pub const TAXON: Taxon = Taxon {
    name: "Auracoccaceae",
    english_name: "Aur Little Spheres",
    description: "These minute aur-fed cells have simple rounded bodies and reproduce by division. \
        Air currents carry the cells between regions of the sky. \
        Their abundance links ambient aur energy to the animals that filter particles from the sky.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
