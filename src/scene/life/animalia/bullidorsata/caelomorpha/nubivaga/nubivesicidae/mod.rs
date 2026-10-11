pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod nubivesica;

pub const TAXON: Taxon = Taxon {
    name: "Nubivesicidae",
    english_name: "Cloud Bladders",
    description: "Light Shapes have rounded, lobed envelopes enclosing tightly packed gas chambers. \
        Their pale jade membranes reflect the surrounding sky, and fine yellow-green veins join the chambers around a basal cluster of gelatinous organs. \
        Short, broad trailing membranes sway beneath the body as the animals drift through high-altitude currents and absorb sparse airborne food.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
