use crate::scene::life::biology::{Biology, Reproduction, Social};

pub const BIOLOGY: Biology = Biology {
    oral_pairs: 3,
    reproduction: Reproduction::LiveBirth,
    social: Social::Herd,
    adult_mass: 12.0,
    perception: 700.0,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod ferrilaminata;
pub mod grandivora;

taxon!(
    Class,
    "Vivipara",
    "Live Bearers",
    [
        "Members of this class bear live young within a protected internal brood chamber.",
        "Three pairs of mouth tentacles evolved from the ancestral tetralate pair, allowing food handling without sacrificing a secure grip or close contact with companions.",
        "Expanded respiratory folds and moisture-recovery chambers support larger bodies and sustained flight.",
        "Gas-storage compartments remain separate from the breathing passages, providing buoyancy without interrupting oxygen exchange.",
        "Prolonged development favours social recognition, vocal communication and moving nursery groups that protect young without a fixed nesting surface.",
    ],
);
