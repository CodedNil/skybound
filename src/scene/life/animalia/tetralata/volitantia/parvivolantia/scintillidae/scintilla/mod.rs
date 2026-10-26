pub use super::{BIOLOGY, render_wings};

pub mod scintillans;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Scintilla",
    "Spark",
    [
        "Members of Scintilla are tiny four-winged herbivores whose bodies emit conspicuous coloured light.",
        "They gather algae and seeds while following the motion of nearby companions.",
        "A swarm remains flexible, opening around obstacles and closing again where food becomes concentrated.",
        "Glowing neighbours remain easier to follow in the dim equatorial skies or when mineral haze blocks distant contrasts.",
        "The same signal that holds the group together also attracts predators, favouring constant collective movement.",
    ],
);
