use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Filter,
    speed: 10.0,
    social: Social::Swarm,
    ..super::BIOLOGY
};

pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod lucerna;

pub const TAXON: Taxon = Taxon {
    name: "Lucernidae",
    english_name: "Lamps",
    description: "Luminous four-winged filter feeders migrate vertically through the sky, gathering suspended nutrients and small luminous prey.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
