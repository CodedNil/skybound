use crate::scene::life::{Rank, Taxon};

pub mod nubiballona;

pub const TAXON: Taxon = Taxon {
    name: "Leviformidae",
    english_name: "Light Shapes",
    description: "Light Shapes have rounded, lobed envelopes enclosing tightly packed gas chambers. \
        Their pale jade membranes reflect the surrounding sky, and fine yellow-green veins join the chambers around a basal cluster of gelatinous organs. \
        Short, broad trailing membranes sway beneath the body as the animals drift through high-altitude currents and absorb sparse airborne food.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
