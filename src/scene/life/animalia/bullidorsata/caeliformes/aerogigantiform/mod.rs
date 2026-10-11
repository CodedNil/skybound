pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod aerogigantidae;

pub const TAXON: Taxon = Taxon {
    name: "Aerogigantiform",
    english_name: "Air Giants",
    description: "Air Giants are enormous golden bubblebacks with broad, vaulted bodies supported by many distributed gas chambers. \
        Branching ribs divide their canopies into rounded chambers and deep arching bays, ending in a scalloped, pointed fringe. \
        They drift through the lower sky, where broad collecting surfaces and moist folds intercept suspended algae and plant matter.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
