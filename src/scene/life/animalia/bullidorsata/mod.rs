use crate::scene::life::{Rank, Taxon};

pub mod caeliformes;

pub const TAXON: Taxon = Taxon {
    name: "Bullidorsata",
    english_name: "Bubblebacks",
    description: "Bubblebacks are basal, sponge-like animals with sticky gelatinous tissues enclosed by buoyant skins. \
        They produce gas in several internal bladders, distributing lift across their bodies and maintaining buoyancy after local damage. \
        Air currents carry them through suspended algae and plant fragments, which adhere to their moist surfaces and are absorbed as food.",
    rank: Rank::Phylum,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
