use crate::scene::life::biology::{Anatomy, Biology, Feeding};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Attached,
        ..Organs::ANIMAL
    },
    anatomy: Anatomy::Suctorial,
    oral_pairs: 1,
    feeding: Feeding::Parasite,
    adult_mass: 0.05,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod errantia;
pub mod volantia;

pub const TAXON: Taxon = Taxon {
    name: "Sanguinivora",
    english_name: "Blood Eaters",
    description: "Blood Eaters descend from soft-bodied, sucker-bearing parasites that obtain fluids from living hosts. \
        Their elongated bodies range from camouflaged, permanently attached leeches to agile winged visitors. \
        Some lineages disguise feeding as grooming; others have evolved mutualistic relationships with their hosts.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
};
