use crate::scene::life::biology::Biology;
pub mod caelomorpha;
pub mod young;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Phylum,
    "Bullidorsata",
    "Bubblebacks",
    [
        "Bubblebacks descend from porous animals whose moist outer tissues both collect and digest suspended food.",
        "Several gas bladders distribute lift through the body, so local damage need not send the whole animal toward the aur ocean.",
        "Gelatinous organs exchange nutrients over short distances without the concentrated digestive crop of active flyers.",
        "Chamber inflation and flexible membranes control orientation while the planetary updraft supplies most movement.",
        "Their descendants divide the sky by arranging the same buoyant and absorptive tissues into compact balloons, trailing leaves or enormous feeding canopies.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
pub mod nature {
    //! The gas-bags have no muscle to fly with and no nerve centre to plan with: they drift, sleep
    //! slack, and breed by releasing spawn among their kind.

    use crate::scene::life::{
        Kind,
        animalia::{
            ETHOLOGY as ANIMAL,
            bullidorsata::young::POLYP,
            mind::{Ethology, Situation},
            physiology::Physiology,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: |_, food| food == Kind::Auracoccus,
        floats: true,
        form: |_, young| if young < 0.7 { POLYP } else { 0 },
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        court: spawn,
        sleep: torpor,
        ..ANIMAL
    };

    /// Spawning: drifting into the thick of its kind to release.
    pub const fn spawn(s: &Situation) -> Goal {
        Goal::new(Activity::Display, s.seen.kin, 0.2)
    }

    /// Torpor: going slack and letting the air carry it.
    pub const fn torpor(_: &Situation) -> Goal {
        Goal::new(Activity::Sleep, None, 0.0)
    }
}
