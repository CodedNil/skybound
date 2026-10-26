pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod scintilla;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.005,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Scintillidae",
    "Sparkies",
    [
        "Members of this family are minute luminous herbivores with two pairs of rapidly controlled wings.",
        "Algae and suspended seeds provide food that can be gathered without attachment to a large plant.",
        "Light and local movement cues keep neighbours together as a swarm spreads across a feeding current.",
        "Small bodies reproduce quickly enough to exploit food concentrations that persist for only a short time.",
        "Their abundance converts microscopic production into prey for larger flyers, especially along the boundaries between feeding streams.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Sparkies are the third lineage to take to swarming, and theirs is held together by
    //! light: the swarm flashes in time, panic runs through it as through the Nibblers', and a
    //! pair courts by matching one another's flashes. A swarm faced with a hunter turns its light
    //! on it, every Sparky flaring at once as they whirl, so the hunter cannot fix on any one;
    //! and in a storm they crowd under the canopy of an Air Giant, out of the rain.

    use crate::scene::life::{
        Kind,
        animalia::{
            SOCIABILITY,
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                display_near, orbit,
            },
            nature::{SHELTER, panic},
            physiology::Physiology,
            tetralata::{ETHOLOGY as TETRALATE, FOLLOW},
        },
        behaviour::{Activity, Goal},
    };

    /// The canopy of an Air Giant, to shelter under.
    const CANOPY: Interest = Interest("canopy");

    /// Whirling round the swarm's centre, all flaring together.
    const DAZZLE: Intent = Intent {
        name: "dazzle",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Display,
                s.seen
                    .kin
                    .map(|kin| kin + orbit(up, kin, 3.0, s.now * 4.0 + s.wobble())),
                1.2,
            )
        },
    };

    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (TETRALATE.notice)(glimpse, seen);
        if !glimpse.heard && glimpse.other == Kind::Aerogigas {
            seen.note(CANOPY, glimpse.offset);
        }
    }

    /// In a storm, in under the nearest canopy if there is one.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == SHELTER
            && let Some(canopy) = s.seen.mark(CANOPY)
        {
            return Some(Goal::new(Activity::Rest, Some(canopy - s.up() * 8.0), 0.9).hovering());
        }
        (TETRALATE.act)(mind, s)
    }

    pub const PHYSIOLOGY: Physiology = Physiology {
        flocks: true,
        speed: 11.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        fear: |s, mind: &mut Mind| (TETRALATE.fear)(s, mind) + panic(s, 1.0),
        court: |s| display_near(s.seen, 1.5, s.seen.kin),
        notice,
        act,
        ..TETRALATE
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.kin.is_some() {
            urges.add(FOLLOW, 0.35 + s.temper(SOCIABILITY) * 0.3);
        }
        if s.seen.kin_count >= 4 && s.seen.threat.is_some() {
            urges.add(DAZZLE, s.threat * 1.5);
        }
    }
}
