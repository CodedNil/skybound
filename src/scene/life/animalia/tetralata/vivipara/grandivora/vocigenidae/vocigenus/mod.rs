pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod antiquus;

pub const TAXON: Taxon = Taxon {
    name: "Vocigenus",
    english_name: "Voice-born One",
    description: "These rare giants carry extensive white fur, including long beard-like growths that hang beneath the head. \
        Their broad wings and unhurried movements suit a diet of large, soft aerial plants. \
        They feed exclusively on kelp and Laevensails, making healthy floating forests essential to their survival.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
