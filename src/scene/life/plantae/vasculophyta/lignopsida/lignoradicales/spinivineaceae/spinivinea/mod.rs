pub use super::{BIOLOGY, render};

pub mod volitans;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Spinivinea",
    "Thorn Vine",
    [
        "Members of Spinivinea grow thorny woody vines into drifting spheres or connected chains.",
        "Broad leaves supply drag while the branching framework yields slightly to changing currents.",
        "Mild auric charge strengthens the defensive role of the thorns without making the whole plant tolerant of concentrated aur.",
        "Interior gaps shelter small animals and retain nutrient-bearing debris.",
        "Chains can spread across neighbouring streams, but must break or rearrange when the difference in airflow exceeds the strength of living connections.",
    ],
);
