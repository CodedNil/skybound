pub use super::render_wings;
use crate::scene::life::biology::Biology;

pub mod herbivolidae;
pub mod vocigenidae;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Order,
    "Grandivora",
    "Large Feeders",
    [
        "Members of this order are large live-bearing flyers adapted to food sources separated by extensive open air.",
        "Broad wings and efficient respiratory folds reduce the cost of carrying a substantial body between meals.",
        "Social recognition and far-carrying calls connect companions even when a feeding group spreads beyond direct contact.",
        "Grazing branches follow algae streams or floating forests, while scavengers exploit carcasses left along the same routes.",
        "Their young are born tiny into a pouch on the mother's back that opens forwards and upwards, so that nothing can fall from it towards the aur; half grown, they make short trips out to feed, longer each time, until they stop coming back.",
        "Prolonged care of young ties their movements to sheltered nursery currents as well as immediate food availability.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Brood, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        brood: Brood::Pouch,
        form: |_, young| if young < 0.2 { 11 } else { 0 },
        ..super::super::PHYSIOLOGY
    };
}
