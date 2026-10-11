use crate::scene::life::biology::{Biology, Reproduction};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Aerosol,
        ..super::BIOLOGY.organs
    },
    adult_mass: 0.002,
    reproduction: Reproduction::Division,
    ..super::BIOLOGY
};

pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod auracoccales;
pub mod luminiflorales;

pub const TAXON: Taxon = Taxon {
    name: "Micrococcopsida",
    english_name: "Small Sphere Forms",
    description: "Small Spheres are microscopic plants that drift freely or form fine growths on larger organisms. \
        Their small size and low mass allow passive suspension in rising air. \
        Rapid reproduction and efficient energy collection make these forms abundant food and important partners or parasites of larger life.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
