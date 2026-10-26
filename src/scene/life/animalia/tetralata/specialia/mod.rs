pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod detritivora;
pub mod rapaciformes;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.2,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Physiology, Take};

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: |_, _| false,
        takes: Take::Carrion,
        gleans: true,
        ..super::super::PHYSIOLOGY
    };
}
