use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 6.0,
    adult_mass: 0.15,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod sericifex;
pub mod sputator;

taxon!(
    Family,
    "Sericificidae",
    "Silk Makers",
    [
        "Members of this family produce adhesive feeding secretions handled by muscular mouth tentacles.",
        "In the silk-building branch, the secretion becomes tensioned strands stretched between independently drifting plants.",
        "More mobile relatives project a sticky acidic form that restrains prey without constructing a lasting trap.",
        "Both branches use controlled wing strokes to compensate for moving anchors and sudden changes in current.",
        "The family flourishes where plants funnel small grazers through narrow feeding passages, especially within the intertwined Jungles.",
    ],
);
