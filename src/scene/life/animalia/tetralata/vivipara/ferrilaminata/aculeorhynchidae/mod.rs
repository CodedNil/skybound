pub use super::{BIOLOGY, render_wings};

pub mod aculeorhynchus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Family,
    "Aculeorhynchidae",
    "Spined Beaks",
    [
        "Members of this family concentrate their hunting weapon along the front of the head.",
        "Long, rigid snouts align with narrow bodies and powerful wings to direct force into a short piercing strike.",
        "Flexible mouth tentacles remain behind the weapon to catch and handle what it has struck.",
        "Their hunting depends on approach angle and a clear recovery path rather than on circling contact with prey.",
        "Open Gardens and the edges of fast wind passages provide the vertical space their body plan requires.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Spined Beaks climb high above their prey, fold their wings and fall on it, driving the
    //! stylet along the dive at whichever of the pump or nerve centre lies nearer its line, with
    //! venom behind it, so a miss still paralyses. The prey falls limp, and the hunter wheels and
    //! races it down to catch it from below and carry it up into rising air to eat. They sleep
    //! gliding on rising air and court spiralling up together.

    use crate::scene::life::{
        animalia::{
            mind::{Aim, Blow, Ethology, Mind, Sighting, Situation, glide_asleep, spiral},
            nature::lead,
            tetralata::{PLAY, vivipara::ETHOLOGY as LIVE_BEARER},
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: stoop,
        court: |s| spiral(s, 5.0),
        sleep: glide_asleep,
        blow: stylet,
        act,
        hunts: true,
        carries: true,
        ..LIVE_BEARER
    };

    /// Play, for needlesharks, is mock stooping: climbing above a fellow and diving past it.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == PLAY
            && let Some(kin) = s.seen.kin
        {
            let up = s.up();
            return Some(if kin.dot(up) > -15.0 {
                Goal::new(Activity::Play, Some(kin + up * 25.0), 0.8)
            } else {
                Goal::new(Activity::Play, Some(kin), 1.8).folded(1.0)
            });
        }
        (LIVE_BEARER.act)(mind, s)
    }

    fn stoop(mind: &mut Mind, s: &Situation, prey: Sighting) -> Goal {
        let up = s.up();
        let to = prey.offset;
        if prey.helpless {
            // Race it down and come up under it; once it is held, climb into rising air to eat.
            return if to.length() < 6.0 {
                Goal::new(Activity::Feed, Some(up * 15.0), 0.4).hovering()
            } else {
                Goal::new(Activity::Feed, Some(to - up * 3.0), 1.5)
            };
        }
        let above = to.dot(up);
        let level = (to - up * above).length();
        if !mind.striking && above > -25.0 {
            // Climb into position high above.
            return Goal::new(Activity::Hunt, Some(to + up * 45.0), 0.8);
        }
        mind.striking = mind.striking || level < 25.0;
        if mind.striking {
            Goal::new(Activity::Strike, Some(lead(s.kind, prey)), 2.2).folded(1.0)
        } else {
            Goal::new(Activity::Hunt, Some(to + up * 45.0), 0.7)
        }
    }

    /// The stylet, driven along the dive at whichever vital organ lies nearer its line, with
    /// venom behind it.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "it fills the blow slot, which some hunters leave empty"
    )]
    fn stylet(aim: &Aim) -> Option<Blow> {
        let entry = aim.toward * aim.extent;
        let target = if (aim.pump - entry).length() < (aim.nerve - entry).length() {
            aim.pump
        } else {
            aim.nerve
        };
        let at = target + aim.scatter;
        let span = aim.extent.length();
        Some(Blow {
            strike: Strike {
                from: at - aim.along * span * 2.0,
                to: at + aim.along * span * 0.3,
                radius: aim.extent.min_element() * 0.05,
                force: 1.2,
                wound: Wound::Pierce,
            },
            stun: 25.0,
            rest: 8.0,
            cost: 0.0,
        })
    }
}
