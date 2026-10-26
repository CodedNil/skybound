pub use super::{BIOLOGY, render_wings};

pub mod crepusculofalx;
pub mod ferrispina;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Ferrispinidae",
    "Iron Spines",
    [
        "Members of this family develop hardened cutting surfaces along wing margins or the spine.",
        "Their flexible bodies and paired wings bring those surfaces across prey during a controlled banking pass.",
        "Camouflaged ambushers use short turns, while larger roaming hunters make wider sweeps and sometimes cooperate.",
        "Feeding tentacles secure the result without turning the cutting surface into a grasping limb.",
        "These related approaches divide the family between mineral haze, open herd routes and the defence of moving nurseries.",
    ],
);
