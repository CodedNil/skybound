pub use super::{BIOLOGY, render_wings};

pub mod errantiformes;

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Class,
    "Errantia",
    "Wanderers",
    [
        "Members of this class retain the soft bodies and attachment organs of the ancestral fluid-feeding lineage.",
        "They lack the powered wings of their more mobile relatives and depend on wind or living carriers for transport.",
        "Concealment protects the unattached stage, while a sucker or internal invasion secures access to food.",
        "Many forms settle permanently once a suitable host is reached, investing in close attachment instead of further travel.",
        "Their distribution follows animal traffic through the Gardens more closely than it follows aur brightness alone.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The wanderers ride their hosts or drift between them: they sleep slack, and breed by
    //! releasing spawn where their kind crowd.

    use crate::scene::life::animalia::{
        bullidorsata::nature::{spawn, torpor},
        mind::Ethology,
        physiology::Physiology,
        sanguinivora::{ETHOLOGY as BLOOD_FEEDER, hosts},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        attaches: true,
        partner: hosts,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        court: spawn,
        sleep: torpor,
        ..BLOOD_FEEDER
    };
}
