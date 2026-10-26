pub use super::{BIOLOGY, render_wings};

pub mod fallax;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Ambigena",
    "Double-natured One",
    [
        "Members of Ambigena are winged fluid feeders that approach larger animals as tolerated grooming companions.",
        "Their rounded bodies and mouth fringes resemble those of unrelated cleaners sharing the same hosts.",
        "Debris removal and antiseptic secretions provide a real benefit while concealing the cost of repeated fluid meals.",
        "The feeding sucker remains inherited from parasitic ancestors, even where the surrounding appendages have become delicate grooming tools.",
        "Busy herd routes favour this mixed relationship because an animal rejected by one host can quickly try another.",
    ],
);
