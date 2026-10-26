pub use super::{BIOLOGY, render};

pub mod laeve;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Laevivelum",
    "Smooth Sail",
    [
        "Members of Laevivelum are broad membrane plants supported chiefly by drag in the updraft.",
        "Flexible tissues spread the aur-collecting surface without the repeated bladder-bearing axis of kelp.",
        "Small adjustments in curvature reduce strain where adjacent air streams move at different speeds.",
        "Overlapping plants create shaded, cooler pockets beneath them despite the diffuse origin of the surrounding light.",
        "Their spores disperse through the current, while damaged margins can remain functional as the rest of the sail continues collecting energy.",
    ],
);
