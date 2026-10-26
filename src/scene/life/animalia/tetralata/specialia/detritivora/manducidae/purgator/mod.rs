pub use super::{BIOLOGY, render_wings};

pub mod mutualis;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Purgator",
    "Cleaner",
    [
        "Members of Purgator are rounded cleaners that feed while hovering close to larger animals.",
        "Their mouth appendages pick parasites and debris from surfaces that shift continually during flight.",
        "Careful approaches and repeated withdrawal give the host control over contact and help establish tolerance.",
        "The cleaner gains concentrated food and access to the sheltered wake of a much larger body.",
        "This dependable relationship supports a different life from the brief feeding swarms of its scavenging relatives.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Sweepers left the swarm to live by cleaning larger animals: they keep their own company,
    //! work the hosts over rather than eating from them, and come close to look at anything new,
    //! the player among them.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Situation, Urges},
            nature::FORAGE,
            tetralata::{
                FOLLOW, INVESTIGATE, specialia::detritivora::manducidae::ETHOLOGY as MANDUCID,
            },
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        act,
        ..MANDUCID
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (MANDUCID.urges)(s, urges);
        urges.scale(FOLLOW, 0.0);
        if s.seen.player.is_some() {
            urges.add(INVESTIGATE, s.temper.curiosity * 0.6);
        }
    }

    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == FORAGE {
            return Some(Goal::new(Activity::Clean, s.seen.food, 0.6).hovering());
        }
        (MANDUCID.act)(mind, s)
    }
}
