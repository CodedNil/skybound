pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod aerohirudo;
pub mod ptilohirudo;

pub const TAXON: Taxon = Taxon {
    name: "Aerohirudinidae",
    english_name: "Air Leeches",
    description: "Flyers combine elongated sucker-bearing bodies with wings suited to intermittent aerial feeding. \
        Some use translucent membranes for hovering ambushes; others glide on feathered surfaces through dim clouds. \
        They grip briefly to take blood or sap and then retreat before a host can respond.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
