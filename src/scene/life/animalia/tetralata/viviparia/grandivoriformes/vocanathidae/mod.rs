pub use super::render_wings;
use crate::scene::life::{Rank, Taxon};

pub mod vocanathus;

pub const TAXON: Taxon = Taxon {
    name: "Vocanathidae",
    english_name: "Born Of Voices",
    description: "Born of Voices are enormous, gentle herbivores associated with slow travel and communication across open air. \
        Broad lifting surfaces support substantial bodies and pale insulating growths. \
        They browse large floating plants, especially kelp and Laevensails.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
