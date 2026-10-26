pub use super::{BIOLOGY, render_wings};

pub mod antiquus;

#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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
    //! farther than any other's. When fed and at ease they gather to sing: a slow wheel of the
    //! great animals, each calling in turn, the chorus rolling out across the sky for hours, and
    //! answered from far off by others of their kind.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges, orbit},
            physiology::Physiology,
            tetralata::{
                INVESTIGATE,
                vivipara::{CALL, ETHOLOGY as LIVE_BEARER, sociable},
            },
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        voice: 30_000.0,
        ..super::super::PHYSIOLOGY
    };

    /// Wheeling slowly with its kind, each calling in turn.
    const CHORUS: Intent = Intent {
        name: "chorus",
        act: |_, s| {
            let up = s.up();
            let centre = s.seen.kin.or_else(|| s.seen.mark(CALL)).unwrap_or_default();
            let turn = ((s.now / 12.0) as u32 + s.individual).is_multiple_of(4);
            let goal = Goal::new(
                Activity::Display,
                Some(centre + orbit(up, centre, 60.0, s.now * 0.05 + s.wobble())),
                0.3,
            );
            if turn { goal.calling() } else { goal }
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
        let content = (1.0 - s.drives.hunger) * (1.0 - s.fear).max(0.0);
        if s.seen.kin.is_some() || s.seen.mark(CALL).is_some() {
            urges.add(CHORUS, 0.4 * content);
        }
    }
}
