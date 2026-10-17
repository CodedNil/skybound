use crate::scene::life::biology::{Biology, Feeding};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Nectar,
    adult_mass: 0.1,
    speed: 5.0,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod nebulaphora;

taxon!(
    Family,
    "Nebulaphoridae",
    "Mistings",
    [
        "Members of this family are gentle nectar feeders with broad translucent membrane wings.",
        "The two wing pairs hold a slow approach beside a flower whose stem may be moving in another current.",
        "Flexible mouth appendages reach nectar while the body brushes pollen-bearing tissues.",
        "Coloured membranes and fine veins remain visible in scattered aur light, helping nearby animals recognise a visiting pollinator.",
        "Their journeys connect isolated flowers across gaps that pollen alone would rarely cross.",
    ],
);
