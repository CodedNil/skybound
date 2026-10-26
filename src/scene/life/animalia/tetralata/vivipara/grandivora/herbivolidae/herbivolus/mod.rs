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

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Situation},
            tetralata::{FOLLOW, vivipara::ETHOLOGY as LIVE_BEARER},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology { act, ..LIVE_BEARER };

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
