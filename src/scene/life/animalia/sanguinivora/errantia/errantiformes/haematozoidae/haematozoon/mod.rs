pub use super::{BIOLOGY, render_wings};

pub mod invasor;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Haematozoon",
    "Blood Animal",
    [
        "Members of Haematozoon are parasites whose feeding and multiplication occur inside a host's circulation.",
        "They retain a short external dispersal stage, linking successive hosts without an adult aerial form.",
        "Surface growths shelter developing juveniles until release into passing currents.",
        "The free stage is compact and soft, conserving material for invasion rather than wings or permanent external armour.",
        "Their success depends on the spacing and contact habits of hosts, making herds and busy plant communities important centres of transmission.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! A Blood Dweller lives its life inside a host's circulation, and its only time in the open
    //! air is the short, desperate one between hosts: a juvenile cast loose has reserves for only a
    //! little while, and makes for the nearest crowd of animals it can find.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges},
            sanguinivora::{ETHOLOGY as WANDERER, HOST},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology { urges, ..WANDERER };

    /// Racing its failing reserves to the nearest host.
    const INFEST: Intent = Intent {
        name: "infest",
        act: |_, s| Goal::new(Activity::Attach, s.seen.mark(HOST), 1.0),
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (WANDERER.urges)(s, urges);
        if s.seen.mark(HOST).is_some() {
            urges.add(INFEST, 0.6 + (1.0 - s.drives.youth) * 0.8);
        }
    }
}
