pub use super::{BIOLOGY, render_wings};

pub mod tacita;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Umbravola",
    "Shadow Flier",
    [
        "Members of Umbravola are intelligent scavengers with broad wings and dark, light-absorbing plumage.",
        "They read changes in wind and temperature to search large volumes of air without continuous powered pursuit.",
        "Calls carry information about food and danger between widely spaced companions.",
        "Mouth tentacles detach and retain loose tissue before a carcass falls into more hazardous lower layers.",
        "Their restrained appearance favours patient searching and cautious feeding rather than the bright social displays of some other large flyers.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Shadowings circle high and wide on rising air for hours, sleeping on the wing, and come down
    //! onto a carcass in slow spirals.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Situation, Urges, glide_asleep, orbit},
            nature::{RIDE, SCAVENGE},
            tetralata::vivipara::{ETHOLOGY as LIVE_BEARER, herd},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        sleep: glide_asleep,
        act,
        ..LIVE_BEARER
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        herd(s, urges);
        if s.seen.updraft < 0.5 {
            urges.add(RIDE, 0.3);
        }
    }

    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == SCAVENGE
            && let Some(carcass) = s.seen.carcass
            && carcass.length() > 40.0
        {
            let up = s.up();
            return Some(Goal::new(
                Activity::Feed,
                Some(carcass + up * 30.0 + orbit(up, carcass, 35.0, s.now * 0.4)),
                0.7,
            ));
        }
        (LIVE_BEARER.act)(mind, s)
    }
}
