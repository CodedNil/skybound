pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod florapollis;
pub mod gigantaflora;

pub const TAXON: Taxon = Taxon {
    name: "Bloomaceae",
    english_name: "Blooms",
    description: "Blooms use luminous or colourful petals and sweet nectar to recruit pollinators from the surrounding sky. \
        Their flowers store aur, creating food for visiting animals and valued healing materials for Næ. \
        Plant size and nectar volume determine whether small Mistings or much larger visitors can feed effectively.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
