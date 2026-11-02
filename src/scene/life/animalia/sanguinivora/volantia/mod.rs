#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ, Part, Tissue};
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
)
.of(Tissue::Muscle);

/// A single hoop of spring fibre round the body at the wing roots.
///
/// It is the only frame a blood-feeder has, and it arose in this class apart from the tetralates'
/// girdles, for the same reason, once muscle had something to beat a wing against.
#[cfg(not(target_arch = "spirv"))]
pub const YOKE: Organ =
    Organ::new("wing yoke", Function::Support, vec3(0.78, 0.70, 0.66)).of(Tissue::Spring);

/// The fliers' wing muscle at the roots of their one pair, anchored on the yoke.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[
    Part::shoulder(WING_MUSCLE, 0, vec3(0.42, 0.5, 0.16)),
    Part::girdle(YOKE, 0, 0.06),
];

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Class,
    "Volantia",
    "Flying Ones",
    [
        "Members of this class are fluid-feeding animals with a single pair of powered wings.",
        "The wings developed within the sucker-bearing lineage, independently of the four-winged body plan of Tetralata, and with them a single hoop of spring fibre round the body at their roots, the class's only frame, which arose apart from the tetralates' girdles for the same work.",
        "Flight allows them to leave a host after feeding rather than depend on permanent attachment.",
        "Grown from folds of the body wall, their wings stay joined to the flank along their whole root and are beaten by the wall's own muscle, so they flap and bow rather than twist as a tetralate's stalked wings do.",
        "Broad membranes favour hovering approaches, while smaller fringed wings suit gliding through cloud cover.",
        "Repeated short visits also provide the behavioural starting point for grooming mimicry and more enduring host partnerships.",
    ],
    anatomy: ANATOMY,
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 14.0,
        ..super::super::PHYSIOLOGY
    };
}
