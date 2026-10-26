pub use super::render_wings;
use crate::scene::life::biology::Biology;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;
pub mod dissimuliformes;
pub mod volucriformes;

/// The muscles that beat the winged blood-feeders' wing sheets: flight arose in them apart from
/// the tetralates, from the body wall rather than at wing roots.
#[cfg(not(target_arch = "spirv"))]
pub const WING_MUSCLE: Organ = Organ::new(
    "wing-sheet muscle",
    Function::Flight,
    vec3(0.60, 0.22, 0.30),
);

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Class,
    "Volantia",
    "Flying Ones",
    [
        "Members of this class are fluid-feeding animals with a single pair of powered wings.",
        "The wings developed within the sucker-bearing lineage, independently of the four-winged body plan of Tetralata.",
        "Flight allows them to leave a host after feeding rather than depend on permanent attachment.",
        "Broad membranes favour hovering approaches, while smaller fringed wings suit gliding through cloud cover.",
        "Repeated short visits also provide the behavioural starting point for grooming mimicry and more enduring host partnerships.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 14.0,
        ..super::super::PHYSIOLOGY
    };
}
