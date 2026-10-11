pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod adherans;
pub mod fluitans;

pub const TAXON: Taxon = Taxon {
    name: "Tenacidae",
    english_name: "Tenacious Ones",
    description: "Tenacious Ones specialise in remaining attached to a host after an initial drifting phase. \
        Low-profile, flexible bodies and strong sucker grips resist airflow and grooming. \
        Camouflage reflects the place of attachment or the vegetation where juveniles wait for a host.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
