pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod fructus;

pub const TAXON: Taxon = Taxon {
    name: "Pomaceae",
    english_name: "Fruits",
    description: "Fruits develop rounded buoyant bodies with thick iridescent rinds that capture aur. \
        Nutrient reserves attract many feeding animals, making clustered plants important feeding grounds. \
        Seeds use light, feathery structures and attachment filaments to leave the parent and reach new currents or hosts.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
