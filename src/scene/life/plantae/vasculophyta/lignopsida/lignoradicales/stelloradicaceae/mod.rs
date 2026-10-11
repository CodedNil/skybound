pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod stelloradix;

pub const TAXON: Taxon = Taxon {
    name: "Stelloradicaceae",
    english_name: "Star Roots",
    description: "Rootlets combine a buoyant upper body with thick roots suspended into nutrient-rich lower air. \
        The roots tolerate limited exposure to aur mists before detaching. \
        Detached tissue can become a dispersal stage, linking the death of an old root to the formation of a new floating plant.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
