pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod crepusculorbis;
pub mod ferrispinae;

pub const TAXON: Taxon = Taxon {
    name: "Ferrivolvidae",
    english_name: "Blade Sweepers",
    description: "Blade Sweepers use sharp claws or spinal blades in sudden aerial attacks. \
        Streamlined bodies and paired wings support fast turns around prey, and some lineages supplement hunting with scavenging. \
        Camouflage, luminous lures and coordinated juvenile groups provide different routes to a successful strike.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
