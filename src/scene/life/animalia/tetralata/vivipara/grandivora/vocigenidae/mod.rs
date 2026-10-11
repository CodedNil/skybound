pub use super::{BIOLOGY, render_wings};
use crate::scene::life::{Rank, Taxon};

pub mod vocigenus;

pub const TAXON: Taxon = Taxon {
    name: "Vocigenidae",
    english_name: "Voice-born Ones",
    description: "Born of Voices are enormous, gentle herbivores associated with slow travel and communication across open air. \
        Broad lifting surfaces support substantial bodies and pale insulating growths. \
        They browse large floating plants, especially kelp and Laevensails.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
