use crate::scene::life::biology::{Biology, Reproduction};

pub const BIOLOGY: Biology = Biology {
    reproduction: Reproduction::Pollination,
    ..super::BIOLOGY
};

pub use super::render;

pub mod polliniflorales;
pub mod pomales;

taxon!(
    Class,
    "Floropsida",
    "Flower Forms",
    [
        "Members of this class are vascular plants that concentrate reproduction into flowers or nutrient-rich fruit bodies.",
        "Flexible channels supply these demanding tissues while buoyant supporting organs keep the reproductive structures in inhabited air.",
        "Flowers attract moving animals with nectar, colour and light, turning visitation into pollen exchange.",
        "Fruit-bearing branches invest instead in reserves and specialised dispersing seeds after fertilisation.",
        "Both strategies connect separated drifting plants without a rooted stage or a dependable fixed distance between parents.",
    ],
);
