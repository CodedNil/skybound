pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod susurraceae;

pub const TAXON: Taxon = Taxon {
    name: "Susurrales",
    english_name: "Whispering",
    description: "Whispering plants direct airflow through hollow woody structures. \
        Their trunks provide lift and stability, while smaller hollow branches turn moving air into varied tones. \
        Associations with luminous sap-feeding animals add active movement to an otherwise wind-supported plant body.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
