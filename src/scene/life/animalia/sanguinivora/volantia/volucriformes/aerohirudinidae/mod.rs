pub use super::{BIOLOGY, render_wings};

pub mod aerohirudo;
pub mod ptilohirudo;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Family,
    "Aerohirudinidae",
    "Air Leeches",
    [
        "Members of this family are winged parasites that make brief feeding visits to animals or plants.",
        "Their elongated bodies retain the ancestral sucker, while one wing pair supplies the control needed to approach and withdraw.",
        "Broad membrane wings favour hovering at feeding grounds, and fringed surfaces favour gliding through dim clouds.",
        "Short meals reduce exposure to defensive movements and avoid the need for permanent host camouflage.",
        "Their ranges follow cover, moisture and host traffic rather than the longest available uninterrupted currents.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::Physiology;

    /// Air leeches make brief visits, letting go of their host once fed.
    pub const PHYSIOLOGY: Physiology = Physiology {
        lets_go: true,
        ..super::super::PHYSIOLOGY
    };
}
