pub use super::{BIOLOGY, render_wings};

pub mod vocigenus;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Vocigenidae",
    "Voice-born Ones",
    [
        "Members of this family are immense herbivores that browse large floating plants during slow journeys.",
        "Broad wings support the body while long insulating fibres protect exposed tissues in cooler currents.",
        "Their mouth tentacles gather soft kelp and sail membranes without the crushing apparatus of armoured grazers.",
        "Complex calls maintain contact between widely separated animals and guide slowly developing young.",
        "Dependence on extensive plant communities makes the continuity of floating forests more important than short-lived concentrations of suspended algae.",
    ],
);
