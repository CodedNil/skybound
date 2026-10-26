pub use super::{BIOLOGY, render_wings};

pub mod membranacea;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Aerohirudo",
    "Air Leech",
    [
        "Members of Aerohirudo are slender fluid feeders with large membrane wings and a muscular sucker.",
        "The wing surfaces favour controlled hovering beside plants and animal feeding routes.",
        "A flexible body lets the mouth remain attached momentarily while the rest of the animal follows a moving host.",
        "Feeding ends before a prolonged grip would provoke vigorous grooming or a defensive turn.",
        "The genus is most successful where many potential hosts pass through sheltered air close to cover.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Air Leeches hover beside the busy feeding routes rather than chase: they hang on their
    //! broad wings off a drinking-plant's edge, where hosts come to them, and dart in for a meal
    //! as one passes.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges},
            sanguinivora::{ETHOLOGY as BLOOD_FEEDER, WATERHOLE},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        ..BLOOD_FEEDER
    };

    /// Hovering off the edge of a drinking-plant.
    const LIE_IN_WAIT: Intent = Intent {
        name: "lie in wait",
        act: |_, s| {
            let up = s.up();
            let beside = s
                .seen
                .mark(WATERHOLE)
                .map(|water| water - water.normalize_or_zero() * 15.0 + up * 4.0);
            Goal::new(Activity::Rest, beside, 0.5).hovering()
        },
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (BLOOD_FEEDER.urges)(s, urges);
        if s.seen.mark(WATERHOLE).is_some() {
            urges.add(LIE_IN_WAIT, 0.3 + s.drives.hunger * 0.3);
        }
    }
}
