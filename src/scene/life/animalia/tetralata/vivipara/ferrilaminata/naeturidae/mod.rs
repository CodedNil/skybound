pub use super::render_wings;
use crate::scene::life::biology::{Biology, Feeding, Organs, Social};
pub mod naeturus;

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        aur_capacity: 1.0,
        ..super::BIOLOGY.organs
    },
    feeding: Feeding::Omnivore,
    social: Social::Mutualist,
    ..super::BIOLOGY
};

taxon!(
    Family,
    "Naeturidae",
    "Nae Kin",
    [
        "Members of this family are the only animals with specialised organs that safely store and use limited aur.",
        "Those organs remain isolated from ordinary nutrient circulation, protecting food-processing tissues from concentrated energy.",
        "The small Næ lineage emphasises fine manipulation, while enormous Nætu provide transport and a shielded living surface.",
        "Both retain the internal brooding and social recognition of their live-bearing ancestors.",
        "Their partnership depends on careful Nætu mist-skimming and mutual care, rather than general immunity to the lethal ocean below.",
    ],
);
