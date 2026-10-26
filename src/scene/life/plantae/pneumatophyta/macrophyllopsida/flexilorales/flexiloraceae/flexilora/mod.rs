pub use super::{BIOLOGY, render};

pub mod spiralis;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Flexilora",
    "Flexible Strap",
    [
        "Members of Flexilora grow flexible tendrils around branches or rough animal surfaces.",
        "Spiralling coils keep several points of contact while leaving long sections free to move in the wind.",
        "Specialised fibres convert repeated stretching and bending into chemical energy.",
        "Fragments can establish new attachments, so ordinary strain also creates a route for dispersal.",
        "Their energy supply follows relative movement between host and air rather than how brightly the sky is lit.",
    ],
);
