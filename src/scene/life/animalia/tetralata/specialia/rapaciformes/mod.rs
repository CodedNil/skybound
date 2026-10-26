pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod sericificidae;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Order,
    "Rapaciformes",
    "Grasping Forms",
    [
        "Members of this order descend from four-winged animals that held small food items with muscular mouth appendages.",
        "Predatory descendants developed adhesive secretions that restrain prey before it can escape into the surrounding air.",
        "Some draw those secretions into suspended strands, while others retain them as projected, chemically active droplets.",
        "Precise hovering and a firm tentacle grip let both forms hunt among moving plants without needing a stable surface.",
        "Their closest shared adaptations concern secretion and prey handling, explaining the relationship between web builders and acid-projecting hunters.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Physiology, Take, hunts};

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: hunts,
        takes: Take::Prey,
        gleans: false,
        ..super::super::PHYSIOLOGY
    };
}
