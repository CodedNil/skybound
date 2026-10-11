pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod aerohirudo;
pub mod ptilohirudo;

pub const TAXON: Taxon = Taxon {
    name: "Voluceridae",
    english_name: "Flyers",
    description: "Flyers combine elongated sucker-bearing bodies with wings suited to intermittent aerial feeding. \
        Some use translucent membranes for hovering ambushes; others glide on feathered surfaces through dim clouds. \
        They grip briefly to take blood or sap and then retreat before a host can respond.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
