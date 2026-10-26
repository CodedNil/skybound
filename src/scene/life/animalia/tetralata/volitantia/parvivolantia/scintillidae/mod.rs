pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod scintilla;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.005,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Scintillidae",
    "Sparkies",
    [
        "Members of this family are minute luminous herbivores with two pairs of rapidly controlled wings.",
        "Algae and suspended seeds provide food that can be gathered without attachment to a large plant.",
        "Light and local movement cues keep neighbours together as a swarm spreads across a feeding current.",
        "Small bodies reproduce quickly enough to exploit food concentrations that persist for only a short time.",
        "Their abundance converts microscopic production into prey for larger flyers, especially along the boundaries between feeding streams.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Sparkies are the third lineage to take to swarming, and theirs is held together by
    //! light: the swarm flashes in time, panic runs through it as through the Nibblers', and a
    //! pair courts by matching one another's flashes.

    use crate::scene::life::animalia::{
        mind::{Ethology, Mind, Situation, Urges, display_near},
        nature::panic,
        physiology::Physiology,
        tetralata::{ETHOLOGY as TETRALATE, FOLLOW},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        flocks: true,
        speed: 11.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        fear: |s, mind: &mut Mind| (TETRALATE.fear)(s, mind) + panic(s, 1.0),
        court: |s| display_near(s.seen, 1.5, s.seen.kin),
        ..TETRALATE
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.kin.is_some() {
            urges.add(FOLLOW, 0.35 + s.temper.sociability * 0.3);
        }
    }
}
