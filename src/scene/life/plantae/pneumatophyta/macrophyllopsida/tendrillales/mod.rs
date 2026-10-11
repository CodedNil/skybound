pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod corallophylaceae;
pub mod spirulaceae;

pub const TAXON: Taxon = Taxon {
    name: "Tendrillales",
    english_name: "Tendril-shaped",
    description: "Tendril-shaped plants use slender, flexible or branching bodies to occupy the surfaces of other lifeforms. \
        Attached spirals and fans exploit motion or ambient aur while their hosts provide support and transport. \
        Their attachment creates intricate living layers on trees, kelp and slow-moving animals.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
