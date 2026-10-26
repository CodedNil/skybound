pub use super::{BIOLOGY, render_wings};

pub mod membranacea;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Aerohirudo",
    "Air Leech",
    [
        "Members of Aerohirudo are slender fluid feeders with large membrane wings and a muscular sucker.",
        "The wing surfaces favour controlled hovering beside plants and animal feeding routes.",
        "A flexible body lets the mouth remain attached momentarily while the rest of the animal follows a moving host.",
        "Feeding ends before a prolonged grip would provoke vigorous grooming or a defensive turn.",
        "The genus is most successful where many potential hosts pass through sheltered air close to cover.",
    ],
);
