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
    //! onto a carcass in slow spirals. They know who kills: a hungry Shadowing that sights a hunter
    //! on the hunt follows it, high above and well behind, and is first to its leavings.

    use crate::scene::life::{
        animalia::{
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                glide_asleep, orbit,
            },
            nature::{RIDE, SCAVENGE},
            tetralata::vivipara::{ETHOLOGY as LIVE_BEARER, herd},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        sleep: glide_asleep,
        act,
        notice,
        ..LIVE_BEARER
    };

    /// A hunter at work.
    const HUNTER: Interest = Interest("hunter");

    /// High above and well behind a hunter, waiting for it to kill.
    const TAIL: Intent = Intent {
        name: "tail",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Drift,
                s.seen
                    .mark(HUNTER)
                    .map(|hunter| hunter - hunter.normalize_or_zero() * 60.0 + up * 50.0),
                0.6,
            )
        },
    };

    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (LIVE_BEARER.notice)(glimpse, seen);
        let hunting = matches!(glimpse.activity, Activity::Hunt | Activity::Strike);
        if !glimpse.heard && hunting && glimpse.other != glimpse.kind {
            seen.note(HUNTER, glimpse.offset);
        }
    }

    fn urges(s: &Situation, urges: &mut Urges) {
        herd(s, urges);
        if s.seen.updraft < 0.5 {
            urges.add(RIDE, 0.3);
        }
        if s.seen.mark(HUNTER).is_some() {
            urges.add(TAIL, s.drives.hunger * 0.9);
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
