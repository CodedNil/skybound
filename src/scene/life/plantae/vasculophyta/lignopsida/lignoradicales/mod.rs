pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod spinivineaceae;
pub mod stelloradicaceae;

pub const TAXON: Taxon = Taxon {
    name: "Lignoradicales",
    english_name: "Woody Roots",
    description: "Woody Roots organise durable stems and nutrient-collecting structures around buoyant aerial bodies. \
        Some form defensive tangled spheres, while others suspend thick roots into lower aur mists. \
        Their slow-changing frameworks provide shelter or exploit nutrients inaccessible to delicate membrane plants.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
