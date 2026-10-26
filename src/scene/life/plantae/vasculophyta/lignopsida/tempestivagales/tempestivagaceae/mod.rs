pub use super::{BIOLOGY, render};

pub mod tempestivaga;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Tempestivagaceae",
    "Storm Wanderers",
    [
        "Members of this family carry large collecting sails around narrow, reinforced woody bodies.",
        "Conductive veins guide captured electrical energy inward instead of allowing an uncontrolled discharge through every tissue.",
        "Changes in sail angle alter drag and the plant's response to surrounding currents.",
        "Sensitivity to electrical changes helps align the collecting surface before a nearby storm reaches its strongest phase.",
        "The family depends on stored charge between storms, allowing movement away from dangerous weather after a successful capture.",
    ],
);
