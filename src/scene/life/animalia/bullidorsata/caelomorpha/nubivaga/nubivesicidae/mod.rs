pub use super::BIOLOGY;

pub mod nubivesica;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Nubivesicidae",
    "Cloud Bladders",
    [
        "Members of this family enclose closely packed gas chambers within rounded, pale jade membranes.",
        "Feeding organs gather near the lower centre, above short trailing folds that catch suspended particles.",
        "The compact envelope exposes less delicate tissue to dry high-altitude air than a long ribbon would.",
        "Fine veins carry absorbed nutrients between the folds and the chamber-lined body.",
        "They favour the moist edges of Cloud Peaks, where buoyancy and water retention matter more than rapid pursuit of food.",
    ],
);
