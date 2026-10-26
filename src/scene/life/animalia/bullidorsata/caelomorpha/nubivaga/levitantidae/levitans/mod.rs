pub use super::BIOLOGY;

pub mod taeniatus;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Genus,
    "Levitans",
    "Levitator",
    [
        "Drifters are flattened bubblebacks with a pointed leading edge and long, flexible feeding ribbons.",
        "Rounded gas chambers support the central organs while the ribbons spread through neighbouring streams of air.",
        "Unequal drag across the body turns it gently toward useful light and food-bearing currents.",
        "A ribbon can fold against the body in rough air, reducing the chance that opposing currents tear it away.",
        "This narrow, yielding form allows them to travel between open Gardens and the lower reaches of the Highlands.",
    ],
);
