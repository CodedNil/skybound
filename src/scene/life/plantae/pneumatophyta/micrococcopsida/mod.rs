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

pub mod auracoccales;
pub mod luminiflorales;

taxon!(
    Class,
    "Micrococcopsida",
    "Small Sphere Forms",
    [
        "Members of this class are small aur-collecting plants with short internal exchange distances.",
        "Free cells rely on the updraft for suspension, while colonial descendants form fine patches on larger organisms.",
        "Rapid division is ancestral, with fragmentation becoming important after growth shifts onto a persistent surface.",
        "Small size allows quick responses to moisture and energy but leaves exposed forms vulnerable to drying.",
        "Their abundance turns fleeting favourable currents and living surfaces into food for a much larger aerial ecosystem.",
    ],
);
