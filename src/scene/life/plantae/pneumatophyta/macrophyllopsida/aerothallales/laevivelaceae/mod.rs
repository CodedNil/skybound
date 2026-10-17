use crate::scene::life::biology::{Biology, Lift, Organs};
pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Membrane,
        ..super::BIOLOGY.organs
    },
    ..super::BIOLOGY
};
pub use super::render;

pub mod lacufolium;
pub mod laevivelum;

taxon!(
    Family,
    "Laevivelaceae",
    "Smooth Sails",
    [
        "Members of this family develop broad, thin membranes from the collecting surfaces of floating frond ancestors.",
        "Drag against rising air carries much of their weight, reducing dependence on bulky gas chambers.",
        "Flat forms shed water and spread widely, while basin-shaped descendants retain rain within a curved rim.",
        "Both must balance collecting area against the extra load imposed by wet tissues and visiting animals.",
        "This shared flexible membrane explains the relationship between open sails and drinking ponds without requiring separate origins for their lifting surfaces.",
    ],
);
