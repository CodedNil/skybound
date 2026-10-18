use crate::scene::life::biology::{Biology, Feeding};
pub mod caelomorpha;

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Absorption,
    speed: 0.0,
    ..super::BIOLOGY
};

taxon!(
    Phylum,
    "Bullidorsata",
    "Bubblebacks",
    [
        "Bubblebacks descend from porous animals whose moist outer tissues both collect and digest suspended food.",
        "Several gas bladders distribute lift through the body, so local damage need not send the whole animal toward the aur ocean.",
        "Gelatinous organs exchange nutrients over short distances without the concentrated digestive crop of active flyers.",
        "Chamber inflation and flexible membranes control orientation while the planetary updraft supplies most movement.",
        "Their descendants divide the sky by arranging the same buoyant and absorptive tissues into compact balloons, trailing leaves or enormous feeding canopies.",
    ],
);
