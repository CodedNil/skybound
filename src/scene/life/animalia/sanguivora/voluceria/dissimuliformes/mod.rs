pub use super::BIOLOGY;
pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod mimicidae;

pub const TAXON: Taxon = Taxon {
    name: "Dissimuliformes",
    english_name: "Disguised Ones",
    description: "Disguised Ones approach hosts by resembling beneficial or harmless visitors. \
        Their soft parasitic bodies are paired with mobile feeding fringes, wing shapes or luminous signals that reduce suspicion. \
        The lineage includes both deceptive fluid feeders and descendants whose relationships have become genuinely mutualistic.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
