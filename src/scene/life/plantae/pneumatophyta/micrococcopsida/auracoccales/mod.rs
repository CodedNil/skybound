pub use super::{BIOLOGY, render};

pub mod auracoccaceae;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Auracoccales",
    "Aur Little Spheres",
    [
        "Members of this order are free-floating microscopic plants sustained by dispersed aur.",
        "Their compact cells remain light enough for suspension without gas bladders or broad membranes.",
        "Water and dissolved minerals pass across the surface, so reproduction follows changes in local air rather than stored reserves.",
        "Rapid division can transform a moist current into a temporary feeding ground.",
        "Their descendants remain distinct from attached luminous patches, whose growth has shifted from free cells to persistent host surfaces.",
    ],
);
