pub use super::{BIOLOGY, render};

pub mod tubulatus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Susurrus",
    "Whisper",
    [
        "Members of Susurrus are hollow-trunked trees that channel rising air through an upright woody body.",
        "Reinforcing ridges maintain the passage while tubular branches divide the airflow into whistles.",
        "A wide lower opening and lighter crown favour stability without a rooted base.",
        "Six major sap veins carry both nutrients and signals to resident Glowgnomes on different sides.",
        "The residents' coordinated movement lets the tree respond to local danger or changing humidity more actively than its lifting structure alone would allow.",
    ],
);
