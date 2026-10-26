pub use super::{BIOLOGY, render_wings};

pub mod venatrix;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Crepusculofalx",
    "Dusk Scythe",
    [
        "Members of Crepusculofalx are camouflaged hunters of the dusty Mineral Fields.",
        "A luminous tail lure draws small animals up into reach of long mouth tentacles, which seize them in a sudden lunge and hold them to the jaws.",
        "The iron they gather from the haze goes not into weapons but into armour: scales along the back and the leading edges of the wings, sparse and uneven, which turn the bites of the larger hunters that come for an animal hanging so still.",
        "Scavenging supplements hunting when mineral dust masks the lure or scatters potential prey.",
        "Their light is most useful against a dim background, linking their hunting success to the local dust load and to the dim equatorial skies they keep to.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Duskslicers hunt by ambush, hanging still in the haze with the lure dangling until prey
    //! comes near, then lunging to seize it in their mouth tentacles and bite it, holding it to
    //! the jaws as they feed. They court by flickering their lures in the dark.

    use crate::scene::life::{
        animalia::{
            mind::{Aim, Blow, Ethology, Mind, Sighting, Situation, display_near, hang_still},
            nature::lead,
            tetralata::vivipara::ferrilaminata::ferrispinidae::ETHOLOGY as IRON_EDGED,
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: ambush,
        court: |s| display_near(s.seen, 6.0, Some(-s.seen.sunward * 100.0)),
        wander: |_, _| hang_still(),
        blow: seize,
        hunts: true,
        carries: true,
        ..IRON_EDGED
    };

    fn ambush(mind: &mut Mind, s: &Situation, prey: Sighting) -> Goal {
        if prey.offset.length() < 18.0 || mind.striking {
            mind.striking = true;
            Goal::new(Activity::Strike, Some(lead(s.kind, prey)), 1.8)
        } else {
            Goal::new(Activity::Hunt, None, 0.0).hovering()
        }
    }

    /// The lunge: the tentacles close round the prey and the jaws bite into it where it was
    /// caught, and it is held, struggling less as the bite takes hold.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "it fills the blow slot, which some hunters leave empty"
    )]
    fn seize(aim: &Aim) -> Option<Blow> {
        let at = aim.toward * aim.extent * 0.6 + aim.scatter * 0.5;
        Some(Blow {
            strike: Strike {
                from: at,
                to: at - aim.toward * aim.extent.min_element() * 0.5,
                radius: aim.extent.min_element() * 0.25,
                force: 0.6 + aim.edge * 0.2,
                wound: Wound::Bite,
            },
            stun: 3.0,
            rest: 2.0,
            cost: 0.0,
        })
    }
}
