pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod adhaerentidae;
pub mod haematozoidae;

pub const TAXON: Taxon = Taxon {
    name: "Errantiformes",
    english_name: "Wanderers",
    description: "Windborne leeches in this order disperse through air currents and attach when they encounter passing animals. \
        Flattened bodies, adhesive feeding surfaces and inconspicuous colours help them survive both before and after attachment. \
        Their reproductive strategies range from external settlement to invasion of a host's internal fluids.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
