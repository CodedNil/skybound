pub use super::{BIOLOGY, render_wings};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::tetralata::SPRING_ROD,
    organs::{Function, Organ, Part, Tissue},
};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;

/// The spring rods of the stone wings, thickened and crusted with the mineral grit they eat with
/// their food, to carry their armour.
#[cfg(not(target_arch = "spirv"))]
pub const STONE_ROD: Organ = Organ::new(
    "grit-crusted rod",
    Function::Support,
    vec3(0.70, 0.64, 0.54),
)
.of(Tissue::Spring);
/// The muscle that works the reinforced mouth against food too tough for the tentacles alone.
#[cfg(not(target_arch = "spirv"))]
pub const CRUSHER: Organ = Organ::new(
    "crushing muscle",
    Function::Handling,
    vec3(0.56, 0.20, 0.18),
)
.of(Tissue::Muscle);

/// The stone wings' frame, heavier for their armour, and the crushing muscle in the head.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[
    Part::rod(
        STONE_ROD,
        vec3(-0.85, 0.6, 0.22),
        vec3(0.95, 0.5, 0.22),
        0.09,
    )
    .replacing(SPRING_ROD),
    Part::lobe(CRUSHER, vec3(-0.95, -0.35, 0.3), vec3(0.3, 0.35, 0.06)),
];

pub mod lithosomatidae;
pub mod ramiphagidae;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Order,
    "Lithoptera",
    "Stone Wings",
    [
        "Members of this order are egg-laying tetralates with reinforced feeding structures and substantial protective coverings.",
        "Broad wings counter the weight of armour, trading speed for controlled movement around resistant vegetation.",
        "Mineral-crusted forms exploit lower root communities, while smaller woody grazers work through Bramble tangles.",
        "Both retain the ancestral pair of mouth appendages, modified to secure food against a dense body.",
        "Their spring rods grow thick and crusted with the mineral grit they swallow, to carry the armour, and a crushing muscle fills the lower head.",
        "Their evolutionary investment in protection and crushing distinguishes them from the lighter particle feeders and pollinators of the same class.",
    ],
    anatomy: ANATOMY,
);
