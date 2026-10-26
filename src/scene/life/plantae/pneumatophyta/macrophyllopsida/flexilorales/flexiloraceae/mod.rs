pub use super::{BIOLOGY, render};

pub mod flexilora;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Flexiloraceae",
    "Flexible Straps",
    [
        "Members of this family specialise in flexible tendrils that gather energy from mechanical deformation.",
        "Their fibres respond to wind-driven stretching, bending and twisting around a living support.",
        "Coiled growth spreads strain across several contact points while leaving collecting lengths free to move.",
        "Fragmentation allows broken pieces to establish on another surface rather than requiring a seed-bearing stage.",
        "They favour active airflow and moving hosts, though the strongest currents can destroy tissue faster than it is renewed.",
    ],
);
