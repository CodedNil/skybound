pub use super::{BIOLOGY, render_wings};
use crate::scene::life::genome::Gene;
pub mod crepusculofalx;
pub mod ferrispina;

/// How long the family's blades grow, crest or wing edge, and how keen: thin, true edges drawn to
/// needle points, or thicker, blunter ones that last longer between honings.
pub const BLADE_LENGTH: Gene = Gene(28);
pub const BLADE_KEENNESS: Gene = Gene(29);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY, WHETSTONE};

taxon!(
    Family,
    "Ferrispinidae",
    "Iron Spines",
    [
        "Members of this family develop hardened cutting surfaces along wing margins or the spine.",
        "Their flexible bodies and paired wings bring those surfaces across prey during a controlled banking pass.",
        "Camouflaged ambushers use short turns, while larger roaming hunters make wider sweeps and sometimes cooperate.",
        "Feeding tentacles secure the result without turning the cutting surface into a grasping limb.",
        "These related approaches divide the family between mineral haze, open herd routes and the defence of moving nurseries.",
        "None is born with its iron: each gathers it all its life, breathing the dust of the mineral haze and scraping it from the stony hides of Grounders, and lays it down a little at a time into its edges and armour, so the old carry far more than the young.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The iron-edged hunters cut with blades of iron-hardened tissue, a crest or a wing's leading
    //! edge, and every cut dulls them a little: a dull edge only bruises. So between hunts they
    //! hone. Each seeks out something harder than itself, the mineral hide of a drifting Grounder
    //! or the woody arms of a Root-Star, and draws its edge along it in long scraping passes,
    //! rolling to work each side in turn. A Grounder bears it without notice; its hide is scored
    //! with the grooves of generations of them.

    use crate::scene::life::{
        Kind,
        animalia::{
            mind::{Ethology, Glimpse, Intent, Interest, Perception, Situation, Urges, orbit},
            physiology::Physiology,
            tetralata::vivipara::ETHOLOGY as LIVE_BEARER,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        ironclad: true,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        notice,
        ..LIVE_BEARER
    };

    /// Something hard enough to hone iron on.
    pub const WHETSTONE: Interest = Interest("whetstone");

    /// Drawing the edge along a whetstone in long rolling passes.
    pub const HONE: Intent = Intent {
        name: "hone",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Hone,
                s.seen
                    .mark(WHETSTONE)
                    .map(|stone| stone + orbit(up, stone, 3.0, s.now * 0.9 + s.wobble())),
                0.5,
            )
        },
    };

    /// A Grounder's stony hide, or a woody plant of some size.
    pub fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (LIVE_BEARER.notice)(glimpse, seen);
        let hard = glimpse.other == Kind::Lithosoma
            || glimpse
                .other
                .plant()
                .is_some_and(|plant| plant.vascular && glimpse.other.biology().adult_mass > 5.0);
        if !glimpse.heard && hard {
            seen.note(WHETSTONE, glimpse.offset);
        }
    }

    /// The blunter the edge, the stronger the pull to hone it, once fed and unafraid.
    pub fn urges(s: &Situation, urges: &mut Urges) {
        (LIVE_BEARER.urges)(s, urges);
        if s.seen.mark(WHETSTONE).is_some() {
            let calm = (1.0 - s.drives.hunger) * (1.0 - s.fear).max(0.0);
            urges.add(HONE, (s.wear * 1.2 + 0.05) * calm);
        }
    }
}
