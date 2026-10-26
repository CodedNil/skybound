pub use super::{BIOLOGY, render_wings};

pub mod acidus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Sputator",
    "Spitter",
    [
        "Members of Sputator use projected adhesive droplets rather than drawn strands to secure prey.",
        "Their secretion has become acidic, helping it penetrate resistant coverings after the initial impact.",
        "Reinforced bodies and compact wings favour a braced strike among vegetation over long pursuit.",
        "The same inherited mouth tentacles that handle silk in related animals stabilise an anchor and gather the immobilised meal.",
        "Their relationship to Weavers is expressed in the feeding secretion and its handling, even though their hunting structures differ.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Spitfires hold off at range, out of reach of a struggling target, and fire the acid their
    //! glands store, which they must wait to refill.

    use crate::scene::life::{
        animalia::{
            mind::{Aim, Blow, Ethology, Mind, Sighting, Situation},
            tetralata::specialia::rapaciformes::sericificidae::ETHOLOGY as SILK_MAKER,
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: snipe,
        blow: acid,
        hunts: true,
        range: 14.0,
        ..SILK_MAKER
    };

    const RANGE: f32 = 12.0;

    fn snipe(_: &mut Mind, _: &Situation, prey: Sighting) -> Goal {
        let to = prey.offset;
        let hold = to - to.normalize_or_zero() * RANGE;
        let activity = if to.length() < RANGE * 1.3 {
            Activity::Strike
        } else {
            Activity::Hunt
        };
        Goal::new(activity, Some(hold), 0.8).hovering()
    }

    /// A gob of acid on the side facing it, if any is left: it ruins the breathing folds and gas
    /// stores under the skin.
    fn acid(aim: &Aim) -> Option<Blow> {
        if aim.reserve < 0.25 {
            return None;
        }
        let at = aim.toward * aim.extent * 0.85 + aim.scatter * 0.3;
        Some(Blow {
            strike: Strike {
                from: at,
                to: at - aim.toward * aim.extent.min_element() * 0.4,
                radius: aim.extent.min_element() * 0.45,
                force: 0.8,
                wound: Wound::Burn,
            },
            stun: 2.0,
            rest: 5.0,
            cost: 0.25,
        })
    }
}
