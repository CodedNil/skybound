use crate::scene::life::biology::{Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Scavenger,
    adult_mass: 0.2,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod detritivora;
pub mod rapaciformes;

taxon!(
    Class,
    "Specialia",
    "Particular Ones",
    [
        "Members of this class are four-winged animals whose feeding is organised around small, scattered or temporarily concentrated resources.",
        "Compact scavengers and cleaners retain organic particles with mobile mouth appendages.",
        "Filter feeders enlarge the collecting surface, while predatory branches use secretions and constructed traps to concentrate living prey.",
        "These specialisations modify an inherited crop, wing arrangement and tentacle pair rather than adding walking limbs.",
        "Their ecological roles connect floating plants and larger animals through nutrient recovery, close contact and the construction of living networks.",
    ],
);
