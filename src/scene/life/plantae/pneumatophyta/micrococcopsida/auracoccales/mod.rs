pub use super::{behave, render};
use crate::scene::life::{Rank, Taxon};

pub mod auracoccaceae;

pub const TAXON: Taxon = Taxon {
    name: "Auracoccales",
    english_name: "Aur Little Spheres",
    description: "Aur Little Spheres are microscopic free-floating plants sustained by collected aur. \
        Rounded cells offer a simple, lightweight body suited to suspension in rising currents. \
        Rapid asexual division turns suitable air into concentrations of primary food for filter feeders.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
