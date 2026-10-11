pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod gigantiflora;
pub mod polliniflora;

pub const TAXON: Taxon = Taxon {
    name: "Pollinifloraceae",
    english_name: "Pollen Flowers",
    description: "Blooms use luminous or colourful petals and sweet nectar to recruit pollinators from the surrounding sky. \
        Their flowers store aur, creating food for visiting animals and valued healing materials for Næ. \
        Plant size and nectar volume determine whether small Mistings or much larger visitors can feed effectively.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
