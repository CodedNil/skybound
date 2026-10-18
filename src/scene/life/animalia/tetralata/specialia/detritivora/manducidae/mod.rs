pub use super::render_wings;
use crate::scene::life::biology::{Biology, Feeding, Social};
pub mod manduca;
pub mod purgator;

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Scavenger,
    social: Social::Swarm,
    speed: 9.0,
    adult_mass: 0.02,
    ..super::BIOLOGY
};

taxon!(
    Family,
    "Manducidae",
    "Nibblers",
    [
        "Members of this family are compact four-winged animals that obtain food from organic debris.",
        "Free-flying scavengers intercept particles in swarms, while cleaners search the surfaces of larger animals.",
        "The same grasping mouth appendages serve either to retain a drifting fragment or to pick material from a host.",
        "Frequent short wing strokes allow precise movement around an unstable food source.",
        "Their feeding returns waste and parasites to the food web before that material is lost into lower, more dangerous air.",
    ],
);
