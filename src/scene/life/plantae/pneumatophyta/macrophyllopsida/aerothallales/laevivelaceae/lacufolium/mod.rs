pub use super::{BIOLOGY, render};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;

pub mod concavum;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

/// The cupped basin that gathers and holds water, which animals drink.
#[cfg(not(target_arch = "spirv"))]
pub const BASIN: Organ = Organ::new("basin", Function::Water, vec3(0.22, 0.46, 0.80));

taxon!(
    Genus,
    "Lacufolium",
    "Lake Leaf",
    [
        "Members of Lacufolium are sail-derived plants whose broad membranes curve into water-holding basins.",
        "A spread rim maintains drag while the centre bears the weight of captured rain.",
        "Flexible curvature allows excess water to escape rather than forcing the whole plant downward.",
        "Animals use the resulting pond, creating a meeting place that brings both protection and the risk of feeding damage.",
        "This water-storing form is most viable in rainy Gardens with dependable lift, rather than the dry upper drifts.",
    ],
);
