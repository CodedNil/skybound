use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 6.0,
    adult_mass: 0.15,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod sericifex;
pub mod sputator;

pub const TAXON: Taxon = Taxon {
    name: "Sericificidae",
    english_name: "Silk Makers",
    description: "Silk-making predators anchor aerial webs between drifting plants. Oral palps manipulate silk, captured prey and nesting material.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
