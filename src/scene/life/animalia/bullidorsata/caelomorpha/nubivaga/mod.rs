pub use super::BIOLOGY;

pub mod levitantidae;
pub mod nubivesicidae;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Nubivaga",
    "Cloud Wanderers",
    [
        "Members of this order are lightweight bubblebacks that feed in open currents and along cloud margins.",
        "Rounded gas chambers hold their gelatinous organs clear of broad, moisture-retaining membranes.",
        "Their feeding surfaces remain exposed to moving air, unlike the sheltered arches of the canopy-bearing lineage.",
        "Compact balloons conserve collecting tissues in sparse air, while ribbon-bearing forms trade protection for a larger feeding reach.",
        "Cloud moisture and changes in updraft strength determine how high these animals can remain without drying or losing lift.",
    ],
);
