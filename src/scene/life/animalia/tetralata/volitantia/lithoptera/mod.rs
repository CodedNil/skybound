pub use super::{BIOLOGY, render_wings};

pub mod lithosomatidae;
pub mod ramiphagidae;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Lithoptera",
    "Stone Wings",
    [
        "Members of this order are egg-laying tetralates with reinforced feeding structures and substantial protective coverings.",
        "Broad wings counter the weight of armour, trading speed for controlled movement around resistant vegetation.",
        "Mineral-crusted forms exploit lower root communities, while smaller woody grazers work through Bramble tangles.",
        "Both retain the ancestral pair of mouth appendages, modified to secure food against a dense body.",
        "Their evolutionary investment in protection and crushing distinguishes them from the lighter particle feeders and pollinators of the same class.",
    ],
);
