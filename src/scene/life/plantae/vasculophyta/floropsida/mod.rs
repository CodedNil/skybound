use crate::scene::life::biology::{Biology, Reproduction};

pub const BIOLOGY: Biology = Biology {
    reproduction: Reproduction::Pollination,
    ..super::BIOLOGY
};

pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod polliniflorales;
pub mod pomales;

pub const TAXON: Taxon = Taxon {
    name: "Floropsida",
    english_name: "Flower Forms",
    description: "Flowering plants concentrate reproduction into conspicuous structures that attract pollinators or disperse seeds. \
        Vascular tissues supply petals, nectar reservoirs and buoyant fruit bodies. \
        Stored aur becomes both an energy reserve and a reward that links these plants to animals and Næ gatherers.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
