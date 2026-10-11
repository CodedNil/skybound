pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod spiralis;

pub const TAXON: Taxon = Taxon {
    name: "Flexilora",
    english_name: "Flexible Strap",
    description: "Cicaia forms long spiralling tendrils attached to trees or animals.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
