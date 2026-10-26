pub use super::{BIOLOGY, render_wings};

pub mod migratorius;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Herbivolus",
    "Plant Flier",
    [
        "Members of Herbivolus are broad-winged herbivores that gather suspended plant food with soft mouth tentacles.",
        "Slow, steady flight keeps the collecting appendages open across a useful portion of the current.",
        "Social spacing lets many animals feed within the same air column without forming a single exhausted wake.",
        "Journeys between clear Verdant Gardens and mineral-rich haze supply different parts of their nutritional needs.",
        "The genus depends on remembered routes and companion recognition rather than the fine pursuit flight of its predatory relatives.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Grazers travel between algae streams in long lines, each falling in behind the one ahead.
    //! A content herd migrates: it keeps to the band of light it lives in and travels along it,
    //! round the pole with the sunset, grazing as it goes, so no stream is cropped to nothing.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Mind, Situation, Urges},
            tetralata::{FOLLOW, vivipara::ETHOLOGY as LIVE_BEARER},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        act,
        ..LIVE_BEARER
    };

    /// Travelling along the band of light, round the pole.
    const MIGRATE: Intent = Intent {
        name: "migrate",
        act: |_, s| {
            let along = s.up().cross(s.seen.sunward).normalize_or_zero();
            Goal::new(
                Activity::Drift,
                Some(s.seen.kin.unwrap_or_default() * 0.3 + along * 200.0),
                0.45,
            )
        },
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (LIVE_BEARER.urges)(s, urges);
        if s.seen.kin_count >= 3 {
            let content = (1.0 - s.drives.hunger * 0.5) * (1.0 - s.fear).max(0.0);
            urges.add(MIGRATE, 0.35 * content);
        }
    }

    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == FOLLOW
            && let Some(kin) = s.seen.kin
        {
            let line = kin - s.seen.heading.normalize_or_zero() * 6.0;
            return Some(Goal::new(
                Activity::Drift,
                Some(line + s.seen.heading * 4.0),
                0.5,
            ));
        }
        (LIVE_BEARER.act)(mind, s)
    }
}
