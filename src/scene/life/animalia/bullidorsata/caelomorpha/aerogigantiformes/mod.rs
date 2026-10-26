pub use super::BIOLOGY;

pub mod aerogigantidae;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Aerogigantiformes",
    "Air Giant Forms",
    [
        "Members of this order are canopy-bearing bubblebacks adapted to feeding across broad columns of rising air.",
        "Their ancestral gas bladders have multiplied into layers supported by flexible branching ribs.",
        "Deep folds beneath the canopy shelter absorptive tissues from drying and sudden changes in airflow.",
        "Rather than chasing food, the animals alter their tilt and collecting area to intercept different portions of the current.",
        "Their immense forms remain tied to productive lower Gardens and descend cautiously as the aur-driven updraft weakens at night.",
    ],
);
