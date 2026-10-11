pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod luminiflora;

pub const TAXON: Taxon = Taxon {
    name: "Luminiferaceae",
    english_name: "Shining Ones",
    description: "Shining Ones grow as bright, low-profile patches on larger organisms. \
        Their tissues gather ambient aur and emit a soft glow. \
        Close contact with a host can support nutrient sharing or competition, depending on the organism beneath them.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
