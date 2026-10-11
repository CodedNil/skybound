pub use super::BIOLOGY;
use crate::scene::life::{Rank, Taxon};

pub mod levitans;

pub const TAXON: Taxon = Taxon {
    name: "Levitantidae",
    english_name: "Levitators",
    description: "Ribbon Shapes have flattened, leaf-like green bodies enclosing several gas chambers. \
        Long, thin membranes extend into curling ribbons, with branching veins and tapered edges that flex in the airflow. \
        The ribbons collect suspended plant matter, while light-sensitive changes in orientation provide gentle directional control through drag.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
};
