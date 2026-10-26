pub use super::{BIOLOGY, render_wings};

pub mod fallax;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Ambigena",
    "Double-natured One",
    [
        "Members of Ambigena are winged fluid feeders that approach larger animals as tolerated grooming companions.",
        "Their rounded bodies and mouth fringes resemble those of unrelated cleaners sharing the same hosts.",
        "Debris removal and antiseptic secretions provide a real benefit while concealing the cost of repeated fluid meals.",
        "The feeding sucker remains inherited from parasitic ancestors, even where the surrounding appendages have become delicate grooming tools.",
        "Busy herd routes favour this mixed relationship because an animal rejected by one host can quickly try another.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Double Natured come as cleaners. A host sees what looks like a Sweeper drifting close
    //! with its fringes working, and holds still to be groomed; only once it is at the skin does
    //! the visitor fasten its sucker, and it cleans honestly as it drinks.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Sighting, Situation},
            nature::lead,
            sanguinivora::ETHOLOGY as BLOOD_FEEDER,
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: visit,
        ..BLOOD_FEEDER
    };

    /// Drifting in slowly as a cleaner would, fastening only at the skin.
    fn visit(_: &mut Mind, s: &Situation, host: Sighting) -> Goal {
        if host.offset.length() > 8.0 {
            Goal::new(Activity::Clean, Some(lead(s.kind, host)), 0.35)
        } else {
            Goal::new(Activity::Attach, Some(host.offset), 0.6)
        }
    }
}
