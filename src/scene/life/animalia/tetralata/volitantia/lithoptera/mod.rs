pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod lithosomatidae;
pub mod ramiphagidae;

pub const TAXON: Taxon = Taxon {
    name: "Lithoptera",
    english_name: "Stone Wings",
    description: "Broad Stone animals have dense bodies, heavy armour and strong feeding structures. \
        Wide wings slow their descent through the updraft. \
        They concentrate around tough plants and mineral-rich currents, where slow movement is compatible with a demanding herbivorous diet.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
