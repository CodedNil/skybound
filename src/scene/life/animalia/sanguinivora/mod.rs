use crate::scene::life::biology::{Anatomy, Biology, Feeding};
use crate::scene::life::biology::{Lift, Organs};

pub const BIOLOGY: Biology = Biology {
    organs: Organs {
        lift: Lift::Attached,
        ..Organs::ANIMAL
    },
    anatomy: Anatomy::Suctorial,
    oral_pairs: 1,
    feeding: Feeding::Parasite,
    adult_mass: 0.05,
    ..super::BIOLOGY
};

pub use super::render_wings;

pub mod errantia;
pub mod volantia;

taxon!(
    Phylum,
    "Sanguinivora",
    "Blood Eaters",
    [
        "Members of this phylum descend from soft-bodied animals with a muscular feeding sucker and paired mouth appendages.",
        "Fluid drawn from living hosts supplies both nutrients and water, allowing the feeding end to serve as an anchor as well as a mouth.",
        "One branch remains wingless and settles for long periods, while another has evolved a single pair of wings for repeated visits.",
        "These wings arose separately from the two ancestral pairs of Tetralata.",
        "Grooming mimicry and plant mutualism developed from brief, restrained feeding, giving some descendants a lasting place in their hosts' lives.",
    ],
);
