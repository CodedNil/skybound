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
        "A luminous tail lure attracts small animals into reach of hardened wing margins.",
        "The body remains quiet in the haze until a short banking movement brings those edges across the target.",
        "Scavenging supplements hunting when mineral dust masks the lure or scatters potential prey.",
        "Their light is most useful against a dim background, linking their hunting success to the local dust load and to the dim equatorial skies they keep to.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Duskslicers hunt by ambush, hanging still in the haze with the lure dangling until prey
    //! comes near, then lunging to scythe it with a wing's iron edge. They court by flickering
    //! their lures in the dark.

    use crate::scene::life::{
        animalia::{
            mind::{Aim, Blow, Ethology, Mind, Sighting, Situation, display_near, hang_still},
            nature::lead,
            tetralata::vivipara::ferrilaminata::ferrispinidae::ETHOLOGY as IRON_EDGED,
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };
    use spirv_std::glam::Vec3;

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: ambush,
        court: |s| display_near(s.seen, 6.0, Some(-s.seen.sunward * 100.0)),
        wander: |_, _| hang_still(),
        blow: scythe,
        hunts: true,
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

    /// The scythe edge of a wing across the flank.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "it fills the blow slot, which some hunters leave empty"
    )]
    fn scythe(aim: &Aim) -> Option<Blow> {
        let side = if aim.toward.x < 0.0 { -1.0 } else { 1.0 };
        let at = Vec3::new(
            side * aim.extent.x * 0.6,
            aim.scatter.y * 0.4,
            aim.scatter.z,
        );
        Some(Blow {
            strike: Strike {
                from: at - aim.along * aim.extent.z * 0.8,
                to: at + aim.along * aim.extent.z * 0.8,
                radius: aim.extent.min_element() * 0.3,
                force: aim.edge,
                wound: Wound::Slash,
            },
            stun: 0.0,
            rest: 4.0,
            cost: 0.0,
        })
    }
}
