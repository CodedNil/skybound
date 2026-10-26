pub use super::{BIOLOGY, render_wings};

pub mod textor;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Sericifex",
    "Silk Maker",
    [
        "Weavers are four-winged predators that draw silk into suspended feeding lines between living plants.",
        "Mouth tentacles adjust anchor points while the wings hold the body clear of vegetation moving in different currents.",
        "Instead of a fixed flat web, they build yielding corridors that intercept small animals approaching leaves, fruit and flowers.",
        "Changes in strand tension reveal both struggling prey and the approach of a wind surge that could tear the structure apart.",
        "Stronger outer lines bind the plant cluster together, creating sheltered sites for eggs and routes used by other inhabitants.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Weavers hunt by waiting. They keep a web strung from a home plant, hang still beside it,
    //! hold fast anything that blunders in, and only then leave it to deliver a venomous bite.
    //! Wind and struggling prey tear a web, so a Weaver's idle hours alternate with mending: it
    //! works slowly round the strands, paying out fresh silk, then hangs still again.

    use crate::scene::life::{
        animalia::{
            PATIENCE,
            mind::{Aim, Blow, Ethology, Hold, Sighting, Snare, hang_still, orbit},
            tetralata::specialia::rapaciformes::sericificidae::ETHOLOGY as SILK_MAKER,
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };
    use spirv_std::glam::Vec3;

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: |_, s, prey| wait(s.seen.home, prey),
        wander: |_, s| {
            // Fed enough not to be watching for prey, it mends; the more patient the weaver,
            // the hungrier it can be and still give its time to the web.
            let mending = s.drives.hunger < 0.25 + s.temper(PATIENCE) * 0.4;
            match s.seen.home {
                Some(web) if mending => Goal::new(
                    Activity::Weave,
                    Some(web + orbit(s.up(), web, 4.0, s.now * 0.3)),
                    0.2,
                )
                .hovering(),
                _ => hang_still(),
            }
        },
        blow: bite,
        snare: web,
        hunts: true,
        anchored: true,
        ..SILK_MAKER
    };

    fn wait(web: Option<Vec3>, prey: Sighting) -> Goal {
        if prey.offset.length() < 15.0 {
            Goal::new(Activity::Strike, Some(prey.offset), 0.6)
        } else {
            Goal::new(Activity::Weave, web, 0.3).hovering()
        }
    }

    /// Anything that strays into the web is held fast.
    fn web(snare: &Snare) -> Option<Hold> {
        snare
            .from_web
            .filter(|&from| from < 10.0 && !snare.fastened)
            .map(|_| Hold {
                stun: 12.0,
                ..Hold::default()
            })
    }

    /// A venomous bite into whatever the web holds.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "it fills the blow slot, which some hunters leave empty"
    )]
    fn bite(aim: &Aim) -> Option<Blow> {
        let at = aim.toward * aim.extent * 0.8;
        Some(Blow {
            strike: Strike {
                from: at,
                to: at * 0.4,
                radius: aim.extent.min_element() * 0.15,
                force: 0.7,
                wound: Wound::Bite,
            },
            stun: 20.0,
            rest: 3.0,
            cost: 0.0,
        })
    }
}
