pub use super::{BIOLOGY, render};

pub mod pendula;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Stelloradix",
    "Star Root",
    [
        "Members of Stelloradix carry star-shaped clusters of fleshy roots below rubbery buoyant crowns.",
        "The roots reach into moist lower currents and supply distant tissues through vascular channels.",
        "Continued aur exposure damages the outer root surfaces, eventually triggering separation before the injury reaches the crown.",
        "Detached roots retain living tissue that curls into a new gas-bearing body.",
        "This cycle couples nutrient gathering to vegetative reproduction while keeping established crowns above the lethal ocean.",
    ],
);
