pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod aeromuscales;

pub const TAXON: Taxon = Taxon {
    name: "Muscopsida",
    english_name: "Moss Forms",
    description: "Moss-like plants form small fuzzy growths that cling to other organisms. \
        Fine leaves and dense surface coverage capture dew and moderate conditions close to the host. \
        Their low profile helps the mats remain attached as their hosts move through the air.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
