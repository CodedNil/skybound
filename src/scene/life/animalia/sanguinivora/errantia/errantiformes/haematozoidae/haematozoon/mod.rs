pub use super::{BIOLOGY, render_wings};

pub mod invasor;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Haematozoon",
    "Blood Animal",
    [
        "Members of Haematozoon are parasites whose feeding and multiplication occur inside a host's circulation.",
        "They retain a short external dispersal stage, linking successive hosts without an adult aerial form.",
        "Surface growths shelter developing juveniles until release into passing currents.",
        "The free stage is compact and soft, conserving material for invasion rather than wings or permanent external armour.",
        "Their success depends on the spacing and contact habits of hosts, making herds and busy plant communities important centres of transmission.",
    ],
);
