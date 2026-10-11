pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod aerogigas;

pub const TAXON: Taxon = Taxon {
    name: "Aerogigantidae",
    english_name: "Air Giants",
    description: "Members of this family have expansive golden canopies with layered gas chambers and a deeply ribbed underside. \
        Fine amber veins and thicker pale-gold ribs form a branching framework around the chambers, while the margins descend into pointed scallops. \
        Gelatinous feeding tissues occupy the sheltered folds and collect plant matter from humid lower-altitude currents.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
