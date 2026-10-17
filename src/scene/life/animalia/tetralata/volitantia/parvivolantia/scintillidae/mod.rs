use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    social: Social::Swarm,
    adult_mass: 0.005,
    speed: 11.0,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod scintilla;

taxon!(
    Family,
    "Scintillidae",
    "Sparkies",
    [
        "Members of this family are minute luminous herbivores with two pairs of rapidly controlled wings.",
        "Algae and suspended seeds provide food that can be gathered without attachment to a large plant.",
        "Light and local movement cues keep neighbours together as a swarm spreads across a feeding current.",
        "Small bodies reproduce quickly enough to exploit food concentrations that persist for only a short time.",
        "Their abundance converts microscopic production into prey for larger flyers, especially along the boundaries between feeding streams.",
    ],
);
