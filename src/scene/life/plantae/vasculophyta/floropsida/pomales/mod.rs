pub use super::{BIOLOGY, render};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;

pub mod pomaceae;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

/// A fruit, which ripens to be eaten and carries its seed away.
#[cfg(not(target_arch = "spirv"))]
pub const FRUIT: Organ = Organ::new("fruit", Function::Fruit, vec3(0.92, 0.66, 0.18));

taxon!(
    Order,
    "Pomales",
    "Fruit-bearing",
    [
        "Members of this order package seeds and stored nutrients inside buoyant reproductive bodies.",
        "Thick collecting rinds help maintain the parent while the interior accumulates food reserves.",
        "Flowers are less conspicuous than the mature fruit, shifting the greatest reproductive investment toward seed provisioning and dispersal.",
        "Ripening releases lightweight seed bodies that respond to currents or attach to passing animals.",
        "Drifting fruit fields consequently serve as feeding grounds while distributing their descendants beyond the air column that supported their growth.",
    ],
);
