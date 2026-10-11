pub use super::{BIOLOGY, render};
use crate::scene::life::{Rank, Taxon};

pub mod versicolor;

pub const TAXON: Taxon = Taxon {
    name: "Chromacorallum",
    english_name: "Coloured Coral",
    description: "Colour Coral forms bright branching growths on kelp and Laevensail trunks and branches.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
