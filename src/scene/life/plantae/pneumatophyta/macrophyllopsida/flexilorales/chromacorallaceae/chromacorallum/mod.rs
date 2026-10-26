pub use super::{BIOLOGY, render};

pub mod versicolor;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Chromacorallum",
    "Coloured Coral",
    [
        "Members of Chromacorallum are branching, aur-collecting plants attached to kelp and sail-bearing vegetation.",
        "Frilled fans extend away from the support, exposing different surfaces as the host turns.",
        "Their varied colours reflect differences in the collecting tissues and remain visible even under diffuse aur light.",
        "A faint glow marks the living margins in dim skies, where the supporting plant is harder to distinguish.",
        "Growth must yield to the host's movement because an attachment that resists every bend would break its own support.",
    ],
);
