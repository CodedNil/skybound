pub use super::BIOLOGY;

pub mod caerulea;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Nubivesica",
    "Cloud Bladder",
    [
        "Members of Nubivesica are compact bubblebacks with rounded chamber-filled bodies and short trailing membranes.",
        "Their organs lie low in the envelope, helping the collecting folds remain beneath the buoyant crown.",
        "Pale translucent skin limits the mass of the outer covering while enclosing moist internal tissues.",
        "Food comes from algae and plant fragments encountered along cloud margins rather than from a broad sheltered feeding canopy.",
        "Their form suits slow travel through thin, humid air where extended appendages would dry rapidly.",
    ],
);
