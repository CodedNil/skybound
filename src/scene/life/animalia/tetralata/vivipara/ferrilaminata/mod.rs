pub use super::render_wings;
use crate::scene::life::biology::{Biology, Feeding};
pub mod aculeorhynchidae;
pub mod ferrispinidae;
pub mod naeturidae;

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Predator,
    speed: 45.0,
    ..super::BIOLOGY
};

taxon!(
    Order,
    "Ferrilaminata",
    "Iron Blades",
    [
        "Members of this order descend from live-bearing four-winged animals adapted to powerful, controlled flight.",
        "Predatory branches specialise the head, wing margins or spine into weapons suited to different aerial strikes.",
        "Their shared internal brooding and prolonged juvenile development also favour recognition and care of companions.",
        "In the Næ lineage, ancestral pursuit and handling abilities have shifted toward manipulation, cooperation and specialised aur protection.",
        "The order therefore retains a common flight and reproductive anatomy despite large differences in diet and adult size.",
    ],
);
