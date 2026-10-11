pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod kelpidales;
pub mod tendrillales;

pub const TAXON: Taxon = Taxon {
    name: "Macrophyllopsida",
    english_name: "Large Leaves",
    description: "Large Leaves spread broad collecting surfaces through the air and distribute lift across multiple gas bladders. \
        Membranes, ribbons and branching fans gather energy while flexible tissues accommodate changing currents. \
        Different growth forms create shade, drinking basins, aerial forests or smaller attached colonies.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
