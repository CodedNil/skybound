pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod naeturae;

pub const TAXON: Taxon = Taxon {
    name: "Næturidae",
    english_name: "",
    description: "This family is the only animal group able to process aur directly. \
        Specialised tissues store aur and use it for healing and protection. \
        The lineage includes small intelligent tool users and huge social carriers whose close mutual dependence shapes both their behaviour and appearance.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
