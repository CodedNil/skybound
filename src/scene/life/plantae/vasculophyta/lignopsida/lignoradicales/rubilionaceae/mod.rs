pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod rubilion;

pub const TAXON: Taxon = Taxon {
    name: "Rubilionaceae",
    english_name: "Thorns",
    description: "Thorns produce woody, defensive frameworks with broad leaves that catch updrafts. \
        Charged spines discourage larger grazers and make the tangled interior useful shelter for smaller animals. \
        Their independently drifting growth can form either compact spheres or longer connected chains.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
