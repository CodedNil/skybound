pub use super::{BIOLOGY, render_wings};

pub mod herbivolus;
pub mod umbravola;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Herbivolidae",
    "Plant Fliers",
    [
        "Members of this family are large, live-bearing animals with broad wings and finely controlled mouth tentacles.",
        "Herbivorous branches gather suspended algae, while scavenging branches search for the remains of larger prey.",
        "Both depend on economical travel between food concentrations scattered through the Gardens.",
        "Sensitivity to airflow and long-distance communication help companions find routes and share feeding opportunities.",
        "Their common flight economy explains the relationship between pale herd grazers and the dark scavengers that follow other animals' journeys.",
    ],
);
