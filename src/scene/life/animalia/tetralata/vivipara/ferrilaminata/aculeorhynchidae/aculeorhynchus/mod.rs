pub use super::{BIOLOGY, render_wings};

pub mod acuminatus;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Aculeorhynchus",
    "Spined Beak",
    [
        "Members of Aculeorhynchus are four-winged predators whose forward feeding structures have become rigid piercing weapons.",
        "A streamlined body aligns behind the head spike during the final approach.",
        "Separate control of the wing pairs steadies a dive and helps recover altitude after the strike.",
        "The inherited mouth tentacles remain flexible and grow long and hooked, catching and holding prey after the rigid weapon has done its work.",
        "This division between striking and handling suits open air where prey cannot be pinned against solid ground.",
    ],
);
