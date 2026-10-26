pub use super::{BIOLOGY, render_wings};

pub mod undulata;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Lucerna",
    "Lamp",
    [
        "Members of Lucerna are long filter feeders with broad wings and softly luminous trailing surfaces.",
        "Their mouth appendages gather suspended organic particles while the wings hold the body across a useful feeding current.",
        "Light attracts small luminous prey, supplementing the sparse material available in open air.",
        "They descend toward richer lower currents to feed and rise again once fed.",
        "These repeated vertical journeys let a single body plan exploit food layers that move as the updraft changes.",
    ],
);
