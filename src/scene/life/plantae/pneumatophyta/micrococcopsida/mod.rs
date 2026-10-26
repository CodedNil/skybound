pub use super::render;
use crate::scene::life::biology::Biology;
pub mod auracoccales;
pub mod luminiflorales;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.002,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation};

    pub const PHYSIOLOGY: Physiology = Physiology {
        floats: false,
        propagation: Propagation::Division,
        ..super::super::PHYSIOLOGY
    };
}
