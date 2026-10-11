pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod venatrix;

pub const TAXON: Taxon = Taxon {
    name: "Crepusculofalx",
    english_name: "Dusk Scythe",
    description: "Duskslicers are medium-sized predators with camouflage suited to the dusty, dim Mineral Fields.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
