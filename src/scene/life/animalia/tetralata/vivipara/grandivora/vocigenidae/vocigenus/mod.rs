pub use super::{BIOLOGY, render_wings};

pub mod antiquus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Vocigenus",
    "Voice-born One",
    [
        "Members of Vocigenus are enormous, slow-moving herbivores with broad wings and extensive white insulating fibres.",
        "Their diet is restricted to kelp and Laevensails, whose large soft tissues reward patient browsing.",
        "Mouth tentacles select and hold vegetation while the body remains clear of the plant's lifting surfaces.",
        "Vocal communication keeps companions connected across gaps between forest communities.",
        "Long development ties the lineage to remembered forest routes, with the polar beams providing bearings across the dim equatorial skies.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Old Wise Ones are curious of anything new in their long lives, and their songs carry
    //! farther than any other's.

    use crate::scene::life::animalia::{
        mind::{Ethology, Situation, Urges},
        tetralata::{
            INVESTIGATE,
            vivipara::{ETHOLOGY as LIVE_BEARER, sociable},
        },
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        ..LIVE_BEARER
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (LIVE_BEARER.urges)(s, urges);
        if s.seen.player.is_some() {
            urges.add(INVESTIGATE, s.temper.curiosity * 0.9);
        }
        // Long-lived and clever, they play and keep company more than most.
        sociable(s, urges, 1.5);
    }
}
