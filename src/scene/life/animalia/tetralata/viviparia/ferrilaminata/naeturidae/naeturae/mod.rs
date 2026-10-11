pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod nae;
pub mod naetu;

pub const TAXON: Taxon = Taxon {
    name: "Næturae",
    english_name: "",
    description: "Aur-adapted relatives in this genus share a close bond between small Næ and enormous Nætu. \
        Feathered tails and paired wings connect very different body sizes, while specialised aur organs enable unusually intimate use of the energy below. \
        Cooperative travel, grooming and protection are central to the association.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
