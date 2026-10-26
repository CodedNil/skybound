pub use super::{BIOLOGY, render_wings};

pub mod adhaerentidae;
pub mod haematozoidae;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Errantiformes",
    "Wanderers",
    [
        "Members of this order are wingless parasites dispersed by wind, vegetation or contact between hosts.",
        "Their sucker-bearing ancestors exchanged active pursuit for secure attachment and concealed feeding.",
        "External forms flatten against skin or wait among kelp, while internal forms develop within host fluids.",
        "The exposed stage remains vulnerable to drying and loss into uninhabited air, favouring release near feeding routes.",
        "Differences in the point of attachment and the duration of dispersal separate the families without requiring independent origins for their feeding organs.",
    ],
);
