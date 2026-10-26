pub use super::render;
use crate::scene::life::biology::Biology;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;
pub mod polliniflorales;
pub mod pomales;

/// A flower, which makes nectar for its pollinators.
#[cfg(not(target_arch = "spirv"))]
pub const BLOOM: Organ = Organ::new("bloom", Function::Nectar, vec3(0.92, 0.45, 0.70));

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Class,
    "Floropsida",
    "Flower Forms",
    [
        "Members of this class are vascular plants that concentrate reproduction into flowers or nutrient-rich fruit bodies.",
        "Flexible channels supply these demanding tissues while buoyant supporting organs keep the reproductive structures in inhabited air.",
        "Flowers attract moving animals with nectar, colour and light, turning visitation into pollen exchange.",
        "Fruit-bearing branches invest instead in reserves and specialised dispersing seeds after fertilisation.",
        "Both strategies connect separated drifting plants without a rooted stage or a dependable fixed distance between parents.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation};

    pub const PHYSIOLOGY: Physiology = Physiology {
        propagation: Propagation::Pollination,
        ..super::super::PHYSIOLOGY
    };
}
