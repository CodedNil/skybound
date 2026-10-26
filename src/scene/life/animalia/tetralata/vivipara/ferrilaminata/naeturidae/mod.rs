pub use super::render_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
use crate::scene::life::{biology::Biology, genome::Gene};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;
pub mod naeturus;

/// A sac that holds aur safely, found only in the Nætu lineage.
#[cfg(not(target_arch = "spirv"))]
pub const AUR_SAC: Organ = Organ::new("aur sac", Function::AurStore, vec3(0.38, 0.85, 0.95));

pub const BIOLOGY: Biology = super::BIOLOGY;

/// How strong the bonds of the Nætu line are: how far one will let a companion or the herd draw
/// away before it calls after them, and how closely it keeps to its rider.
pub const LOYALTY: Gene = Gene(28);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Naeturidae",
    "Nae Kin",
    [
        "Members of this family are the only animals with specialised organs that safely store and use limited aur.",
        "Those organs remain isolated from ordinary nutrient circulation, protecting food-processing tissues from concentrated energy.",
        "The small Næ lineage emphasises fine manipulation, while enormous Nætu provide transport and a shielded living surface.",
        "Both retain the internal brooding and social recognition of their live-bearing ancestors.",
        "Their partnership depends on careful Nætu mist-skimming and mutual care, rather than general immunity to the lethal ocean below.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Nætu lineage is curious and playful: the young chase one another, and anything new is
    //! worth a closer look.

    use crate::scene::life::animalia::{
        CURIOSITY,
        mind::{Ethology, Situation, Urges},
        physiology::{Physiology, Take, omnivore},
        tetralata::{
            INVESTIGATE,
            vivipara::{ETHOLOGY as LIVE_BEARER, sociable},
        },
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: omnivore,
        takes: Take::Living,
        aur: 1.0,
        flocks: false,
        partner: |kind, other| other != kind && other.taxon().descends_from(&super::TAXON),
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        ..LIVE_BEARER
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (LIVE_BEARER.urges)(s, urges);
        if s.seen.player.is_some() {
            urges.add(INVESTIGATE, s.temper(CURIOSITY) * 0.9);
        }
        // The cleverest of all, and the most playful and sociable.
        sociable(s, urges, 1.5);
    }
}
