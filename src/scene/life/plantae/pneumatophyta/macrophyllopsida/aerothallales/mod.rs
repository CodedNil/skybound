pub use super::{BIOLOGY, render};

pub mod aerothallaceae;
pub mod laevivelaceae;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Aerothallales",
    "Air Fronds",
    [
        "Members of this order expand aur-collecting membranes around flexible supporting tissues.",
        "Distributed gas chambers support the long axes of kelp, while sail-derived forms rely more heavily on drag across a wide surface.",
        "The same yielding tissues let related plants form ribbon forests, horizontal sheets or shallow rain basins.",
        "Water and dissolved nutrients are gathered across exposed surfaces rather than from a terrestrial root system.",
        "Their habitats emerge from the balance between lift, rain load and flexibility in currents that differ across the length of a single plant.",
    ],
);
