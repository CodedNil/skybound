pub use super::render_wings;
use crate::scene::life::biology::{Biology, Organs};
pub mod dissimuliformes;
pub mod volucriformes;

pub const BIOLOGY: Biology = Biology {
    organs: Organs::ANIMAL,
    wing_pairs: 1,
    speed: 14.0,
    ..super::BIOLOGY
};

taxon!(
    Class,
    "Volantia",
    "Flying Ones",
    [
        "Members of this class are fluid-feeding animals with a single pair of powered wings.",
        "The wings developed within the sucker-bearing lineage, independently of the four-winged body plan of Tetralata.",
        "Flight allows them to leave a host after feeding rather than depend on permanent attachment.",
        "Broad membranes favour hovering approaches, while smaller fringed wings suit gliding through cloud cover.",
        "Repeated short visits also provide the behavioural starting point for grooming mimicry and more enduring host partnerships.",
    ],
);
