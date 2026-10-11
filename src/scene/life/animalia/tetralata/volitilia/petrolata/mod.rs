pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod ramiphagae;
pub mod terrapodae;

pub const TAXON: Taxon = Taxon {
    name: "Petrolata",
    english_name: "Broad Stone",
    description: "Broad Stone animals have dense bodies, heavy armour and strong feeding structures. \
        Wide wings slow their descent through the updraft. \
        They concentrate around tough plants and mineral-rich currents, where slow movement is compatible with a demanding herbivorous diet.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
