pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod tubulatus;

pub const TAXON: Taxon = Taxon {
    name: "Susurrus",
    english_name: "Whisper",
    description: "Whisperwood is a rare tree with a huge upright hollow trunk, wider at the base, that channels air for lift and stability.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
