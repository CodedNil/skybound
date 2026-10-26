pub use super::BIOLOGY;

pub mod communis;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Auracoccus",
    "Aur Sphere",
    [
        "Members of Auracoccus are microscopic plants suspended freely in the updraft.",
        "Rounded cells exchange water and nutrients across their entire surface while collecting dispersed aur.",
        "Asexual division responds quickly to favourable moisture and energy, producing short-lived feeding concentrations.",
        "Currents distribute the cells between communities, while collecting membranes remove many before they travel far.",
        "Their simple body plan succeeds through abundance and rapid renewal rather than protection from the lethal ocean below.",
    ],
);
