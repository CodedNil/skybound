pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod lithoptera;
pub mod parvivolantia;

pub const TAXON: Taxon = Taxon {
    name: "Volitantia",
    english_name: "Flutterers",
    description: "Flutterers retain conspicuous paired wings and comparatively simple aerial feeding strategies. \
        Small forms flutter between food particles and flowers, while dense relatives use broad surfaces to slow their descent. \
        Their appearance ranges from brilliant light and stained-glass membranes to mineral armour.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
};
