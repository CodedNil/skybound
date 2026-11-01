pub use super::{BIOLOGY, render};

pub mod vittatus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Aerothallus",
    "Air Frond",
    [
        "Members of Aerothallus grow long fibrous strands supported by many spaced gas bladders.",
        "Fans of fine charged hairs hang from the nodes to gather the dispersed aur drifting up from below, and a spiral vane winds each strand to turn the updraft into lift.",
        "Distributed lift keeps one damaged bladder from collapsing the entire strand toward the ocean.",
        "Branches and neighbouring plants can interweave into three-dimensional forests without any rooted foundation.",
        "These forests provide food, shelter and anchor sites, but must remain flexible enough to survive shifts in the currents carrying them.",
    ],
);
