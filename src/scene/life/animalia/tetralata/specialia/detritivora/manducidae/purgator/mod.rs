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
    //! the player among them. A Sweeper advertises: when a large animal passes, it rises to meet it
    //! in a bobbing, side-to-side dance that every animal of the sky has learnt means a cleaner.

    use crate::scene::life::{
        animalia::{
            CURIOSITY,
            mind::{Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges},
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
        notice,
        ..MANDUCID
    };

    /// A large animal that might want cleaning.
    const CLIENT: Interest = Interest("client");

    /// The cleaner's dance: bobbing and swaying before a client.
    const ADVERTISE: Intent = Intent {
        name: "advertise",
        act: |_, s| {
            let up = s.up();
            let sway = up
                .cross(s.seen.mark(CLIENT).unwrap_or(up))
                .normalize_or_zero();
            let dance = up * (s.now * 3.0).sin() * 1.5 + sway * (s.now * 1.7).sin() * 2.0;
            Goal::new(
                Activity::Display,
                s.seen
                    .mark(CLIENT)
                    .map(|client| client - client.normalize_or_zero() * 8.0 + dance),
                0.7,
            )
            .hovering()
        },
    };

    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (MANDUCID.notice)(glimpse, seen);
        let large = glimpse
            .other
            .animal()
            .is_some_and(|animal| !animal.parasite)
            && glimpse.other.biology().adult_mass > glimpse.kind.biology().adult_mass * 50.0;
        if !glimpse.heard && large {
            seen.note(CLIENT, glimpse.offset);
        }
    }

    fn urges(s: &Situation, urges: &mut Urges) {
        (MANDUCID.urges)(s, urges);
        urges.scale(FOLLOW, 0.0);
        if s.seen.player.is_some() {
            urges.add(INVESTIGATE, s.temper(CURIOSITY) * 0.6);
        }
        if s.seen
            .mark(CLIENT)
            .is_some_and(|client| client.length() < 80.0)
        {
            urges.add(ADVERTISE, 0.35 + s.drives.hunger * 0.5);
        }
    }

    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == FORAGE {
            return Some(Goal::new(Activity::Clean, s.seen.food, 0.6).hovering());
        }
        (MANDUCID.act)(mind, s)
    }
}
