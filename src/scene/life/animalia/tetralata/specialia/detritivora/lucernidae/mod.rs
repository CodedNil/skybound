use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Filter,
    speed: 10.0,
    social: Social::Swarm,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod lucerna;

taxon!(
    Family,
    "Lucernidae",
    "Lamps",
    [
        "Members of this family are elongated, luminous filter feeders descended from small four-winged scavengers.",
        "Broad wings and trailing tissues expand the volume of air sampled during a slow feeding pass.",
        "Mouth appendages gather microscopic organic food, while emitted light attracts small prey into the same flow.",
        "Vertical migration follows changes in food concentration and updraft strength through the aur cycle.",
        "Their lower feeding journeys approach the Veilands without granting the tolerance of concentrated aur found in Nætu.",
    ],
);
