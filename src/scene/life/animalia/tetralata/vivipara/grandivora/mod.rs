pub mod herbivolidae;
pub mod vocigenidae;

/// A second pair of spring rods slung low along the belly, which carries the weight of a gut and
/// a brood too heavy for the back's rods alone.
#[cfg(not(target_arch = "spirv"))]
pub const SLING: Organ =
    Organ::new("belly sling", Function::Support, vec3(0.86, 0.80, 0.70)).of(Tissue::Spring);

/// The heavy feeders' frame, slung underneath as well as along the back.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[Part::rod(
    SLING,
    vec3(-0.5, -0.72, 0.3),
    vec3(0.75, -0.65, 0.3),
    0.05,
)];

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Order,
    "Grandivora",
    "Large Feeders",
    [
        "Members of this order are large live-bearing flyers adapted to food sources separated by extensive open air.",
        "Broad wings and efficient respiratory folds reduce the cost of carrying a substantial body between meals, and a second pair of spring rods slung along the belly carries the weight of the gut and the brood.",
        "Social recognition and far-carrying calls connect companions even when a feeding group spreads beyond direct contact.",
        "Grazing branches follow algae streams or floating forests, while scavengers exploit carcasses left along the same routes.",
        "Their young are born tiny into a pouch on the mother's back that opens forwards and upwards, so that nothing can fall from it towards the aur; half grown, they make short trips out to feed, longer each time, until they stop coming back.",
        "Prolonged care of young ties their movements to sheltered nursery currents as well as immediate food availability.",
    ],
    anatomy: ANATOMY,
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Brood, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        brood: Brood::Pouch,
        ..super::super::PHYSIOLOGY
    };
}
