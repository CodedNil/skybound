pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod flexilora;

pub const TAXON: Taxon = Taxon {
    name: "Spirulaceae",
    english_name: "Spirals",
    description: "Spirals attach flexible tendrils to trees or animals and convert repeated wind-driven deformation into chemical energy. \
        Specialised fibres respond to stretching, bending and twisting. \
        Broken pieces can establish new growth, making fragmentation both a means of dispersal and a natural consequence of their exposed habitat.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
