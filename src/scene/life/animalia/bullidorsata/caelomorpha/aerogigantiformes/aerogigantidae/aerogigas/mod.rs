pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod auratus;

pub const TAXON: Taxon = Taxon {
    name: "Aerogigas",
    english_name: "Air Giant",
    description: "Air Giants are enormous golden drifters with a broad, gently domed canopy and a deeply vaulted underside.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
