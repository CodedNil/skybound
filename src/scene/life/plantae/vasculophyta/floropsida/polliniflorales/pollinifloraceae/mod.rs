pub use super::{BIOLOGY, render};

pub mod gigantiflora;
pub mod polliniflora;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Pollinifloraceae",
    "Pollen Flowers",
    [
        "Members of this family use nectar-filled flowers to recruit animals as pollen carriers.",
        "Layered petals and luminous tissues present signals that remain useful through changing aur brightness.",
        "Buoyant bodies and flexible vascular stems support the weight of the flower and temporary visitors.",
        "Small forms reward frequent visits, while giant forms supply enough nectar for larger travellers.",
        "Stored aur also gives some tissues healing value to Næ, connecting reproduction to careful gathering as well as animal feeding.",
    ],
);
