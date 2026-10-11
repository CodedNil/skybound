pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod crepusculofalx;
pub mod ferrispina;

pub const TAXON: Taxon = Taxon {
    name: "Ferrispinidae",
    english_name: "Iron Spines",
    description: "Blade Sweepers use hardened wing margins or spinal blades in sudden aerial attacks. \
        Streamlined bodies and paired wings support fast turns around prey, and some lineages supplement hunting with scavenging. \
        Camouflage, luminous lures and coordinated juvenile groups provide different routes to a successful strike.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
