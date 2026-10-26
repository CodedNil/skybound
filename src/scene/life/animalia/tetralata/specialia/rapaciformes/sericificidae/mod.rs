pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod sericifex;
pub mod sputator;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.15,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Sericificidae",
    "Silk Makers",
    [
        "Members of this family produce adhesive feeding secretions handled by muscular mouth tentacles.",
        "In the silk-building branch, the secretion becomes tensioned strands stretched between independently drifting plants.",
        "More mobile relatives project a sticky acidic form that restrains prey without constructing a lasting trap.",
        "Both branches use controlled wing strokes to compensate for moving anchors and sudden changes in current.",
        "The family flourishes where plants funnel small grazers through narrow feeding passages, especially within the intertwined Jungles.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! In the silk-makers, courtship is a gift: a suitor brings its chosen mate food it has caught.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Situation},
            physiology::Physiology,
            tetralata::ETHOLOGY as TETRALATE,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 6.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        court: gift,
        ..TETRALATE
    };

    fn gift(s: &Situation) -> Goal {
        Goal::new(Activity::Display, s.seen.mate.map(|(mate, _)| mate), 0.5)
    }
}
