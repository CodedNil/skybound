pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod lucerna;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Lucernidae",
    "Lamps",
    [
        "Members of this family are elongated, luminous filter feeders descended from small four-winged scavengers.",
        "Broad wings and trailing tissues expand the volume of air sampled during a slow feeding pass.",
        "Mouth appendages gather microscopic organic food, while emitted light attracts small prey into the same flow.",
        "Vertical migration follows changes in food concentration and updraft strength.",
        "Their lower feeding journeys approach the Veilands without granting the tolerance of concentrated aur found in Nætu.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Shimmerays swarm too, in their own way: a slow ribbon of light drifting together,
    //! sinking to the rich air above the mists to feed and rising again once fed, and courting
    //! by the glow of their lures. A hungry Shimmeray that sees a swarm of Sparkies goes still and
    //! answers their flashes with its lure until they come to it.

    use crate::scene::life::{
        Kind,
        animalia::{
            SOCIABILITY,
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges,
                display_near,
            },
            physiology::{Physiology, Take},
            tetralata::{ETHOLOGY as TETRALATE, FOLLOW},
        },
        behaviour::{Activity, Goal},
    };
    use spirv_std::glam::Vec3;

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: |_, food| matches!(food, Kind::Auracoccus | Kind::Scintilla),
        takes: Take::Living,
        gleans: false,
        speed: 10.0,
        flocks: true,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        court: |s| display_near(s.seen, 4.0, s.seen.kin),
        wander,
        notice,
        ..TETRALATE
    };

    /// A swarm of Sparkies, which a hungry Shimmeray can lure.
    const SPARKS: Interest = Interest("sparks");

    /// Hanging dead still, the lure glowing a Sparky's own flash back at it, until the swarm comes
    /// close enough to sweep in.
    const LURE: Intent = Intent {
        name: "lure",
        act: |_, _| Goal::new(Activity::Display, None, 0.0).hovering(),
    };

    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (TETRALATE.notice)(glimpse, seen);
        if !glimpse.heard && glimpse.other == Kind::Scintilla {
            seen.note(SPARKS, glimpse.offset);
        }
    }

    fn urges(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.kin.is_some() {
            urges.add(FOLLOW, 0.3 + s.temper(SOCIABILITY) * 0.2);
        }
        // Sparkies too far to sweep are drawn in by the lure instead.
        if s.seen
            .mark(SPARKS)
            .is_some_and(|sparks| sparks.length() > 15.0)
        {
            urges.add(LURE, s.drives.hunger * 0.9);
        }
    }

    /// Down to the rich air near the mists when hungry, up when fed.
    fn wander(mind: &mut Mind, s: &Situation) -> Goal {
        let rise = s.up() * (0.5 - mind.hunger) * 60.0;
        Goal::new(
            Activity::Drift,
            Some(s.seen.kin.unwrap_or(Vec3::ZERO) + rise),
            0.25,
        )
    }
}
