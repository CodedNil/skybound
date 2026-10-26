pub use super::{BIOLOGY, render_wings};

pub mod gregaria;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Manduca",
    "Muncher",
    [
        "Members of Manduca are small, agile scavengers that feed in coordinated aerial swarms.",
        "Their inherited mouth tentacles hold fragments while rapid wing strokes maintain position beside moving food.",
        "Different wing pairs can adjust the body through a turn without requiring a perch.",
        "Swarms gather around temporary debris concentrations and disperse when the current dilutes them.",
        "This opportunistic feeding links brief events such as grazing damage to the wider recycling of nutrients.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Nibblers court in pairs spiralling up out of the swarm, and the young chase one another
    //! through it.

    use crate::scene::life::animalia::{
        mind::{Ethology, Situation, Urges, spiral},
        tetralata::{PLAY, specialia::detritivora::manducidae::ETHOLOGY as MANDUCID},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        court: |s| spiral(s, 3.0),
        ..MANDUCID
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (MANDUCID.urges)(s, urges);
        if s.drives.youth < 0.8 && s.seen.kin_count > 0 {
            urges.add(PLAY, (1.0 - s.drives.youth) * 0.5 * (1.0 - s.drives.hunger));
        }
    }
}
