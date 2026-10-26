pub use super::{BIOLOGY, render};

pub mod macrophyllopsida;
pub mod micrococcopsida;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Phylum,
    "Pneumatophyta",
    "Air Plants",
    [
        "Members of this phylum remain aerial through low mass, broad collecting surfaces, internal gas or attachment to a carrier.",
        "Their energy-collecting tissues meet the air directly rather than depending on woody roots supplied from a fixed substrate.",
        "Microscopic branches divide freely, while larger branches organise those tissues into flexible ribbons, membranes and fans.",
        "Gas bladders and drag surfaces distribute support through the larger forms, with attached descendants reducing those lifting structures.",
        "Their evolution produces extensive floating habitats from a light, flexible foundation distinct from the vascular networks of woody and flowering plants.",
    ],
);
