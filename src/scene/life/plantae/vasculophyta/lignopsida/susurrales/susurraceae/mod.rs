pub use super::{BIOLOGY, render};

pub mod susurrus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Susurraceae",
    "Whispers",
    [
        "Members of this family grow ridged hollow trunks and tubular branches with delicate fan leaves.",
        "Airflow through branch openings produces a changing whistle pattern that reflects both anatomy and current strength.",
        "Sap channels supply tissues around the hollow core without filling the lifting passage.",
        "Resident Glowgnomes use the sheltered wood and respond to chemical signals carried through those channels.",
        "Their combined wing force helps the tree adjust its position, making a living partnership part of the family's control of drift.",
    ],
);
