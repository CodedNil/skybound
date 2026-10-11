pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod parvifuga;
pub mod petrolata;

pub const TAXON: Taxon = Taxon {
    name: "Volitilia",
    english_name: "Flutterers",
    description: "Flutterers retain conspicuous paired wings and comparatively simple aerial feeding strategies. \
        Small forms flutter between food particles and flowers, while dense relatives use broad surfaces to slow their descent. \
        Their appearance ranges from brilliant light and stained-glass membranes to mineral armour.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
