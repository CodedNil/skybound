pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod haematozoon;

pub const TAXON: Taxon = Taxon {
    name: "Haematozoidae",
    english_name: "Blood Animals",
    description: "These parasites use a host's internal fluids as both food and developmental habitat. \
        Their soft bodies and specialised juvenile stages support invasion, growth and eventual emergence. \
        The visible external animal is only part of a life cycle that links successive hosts.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
