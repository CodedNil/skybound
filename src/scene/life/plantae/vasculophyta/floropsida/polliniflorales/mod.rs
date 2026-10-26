pub use super::{BIOLOGY, render};

pub mod pollinifloraceae;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Polliniflorales",
    "Pollen Flowers",
    [
        "Members of this order hold nectar-bearing flowers in the current on buoyant vascular bodies.",
        "Petals and luminous tissues advertise a concentrated food reward against the diffuse aur-lit sky.",
        "Flexible stems let a flower turn under a visitor's weight without breaking its supply channels.",
        "Contact transfers pollen between plants whose independent drift makes direct reproductive contact unreliable.",
        "Differences in flower scale divide the order between frequent small visitors and larger animals requiring substantial nectar stores.",
    ],
);
