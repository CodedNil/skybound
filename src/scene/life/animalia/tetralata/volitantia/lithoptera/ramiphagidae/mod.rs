use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 3.0,
    adult_mass: 2.0,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod ramiphaga;

taxon!(
    Family,
    "Ramiphagidae",
    "Branch Eaters",
    [
        "Members of this family are armoured herbivores with reinforced mouth structures for woody plant tissue.",
        "Compact wings hold the body close to a drifting framework while the feeding appendages secure a short-lived grip.",
        "Tough outer coverings reduce damage from charged thorns and scraping branches.",
        "Their feeding opens plant material that softer grazers cannot use, releasing nutrients through fragments and waste.",
        "Dependence on woody vegetation makes dense Bramble communities more suitable than open columns of suspended algae.",
    ],
);
