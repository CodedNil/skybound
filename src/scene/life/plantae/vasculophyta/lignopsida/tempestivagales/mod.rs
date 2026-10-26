pub use super::{BIOLOGY, render};

pub mod tempestivagaceae;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Tempestivagales",
    "Storm Wanderers",
    [
        "Members of this order are woody vascular plants specialised for electrical energy in storm-bearing air.",
        "Collecting veins connect exposed membranes to storage tissues protected within the supporting body.",
        "Broad sails alter drag so that changing currents carry the plant toward or away from a developing storm.",
        "This energy source rewards intermittent charging rather than constant exposure to the brightest aur-lit air.",
        "Their strengthened structure shares the foundation of other woody plants, while their collecting tissues and drift responses define a separate storm-adapted branch.",
    ],
);
