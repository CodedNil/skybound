pub use super::{BIOLOGY, render_wings};

pub mod adhaerens;
pub mod fluitans;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Adhaerentidae",
    "Clingers",
    [
        "Members of this family are flattened parasites that remain attached after finding a host.",
        "Flexible bodies and strong suckers resist both airflow and the movements of an animal in flight.",
        "Their young reach hosts either by drifting freely or by waiting among vegetation visited by grazers.",
        "Camouflage develops around the surface on which each form waits or feeds.",
        "Slow feeding and concealment reduce removal, but specialised cleaners keep these parasites from occupying every suitable patch of skin.",
    ],
);
