pub use super::{BIOLOGY, render_wings};

pub mod pennata;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Ptilohirudo",
    "Feathered Leech",
    [
        "Members of Ptilohirudo are small parasites with fringed wings suited to gliding through cloud layers.",
        "Fine wing fibres spread the lifting surface without the mass of a broad, heavily supported membrane.",
        "Wide-spectrum vision helps them locate silhouettes and aur-lit movement in poor visibility.",
        "They emerge from cloud cover for a short fluid meal and retreat before the host can pursue them.",
        "Their compact form suits the Cloud Peaks, where moisture is abundant but exposed air offers little food or shelter.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Feathered Leeches glide the cloud margins on their ring of a wing, and sleep on it.

    use crate::scene::life::animalia::{
        mind::{Ethology, glide_asleep},
        sanguinivora::ETHOLOGY as BLOOD_FEEDER,
    };

    pub const ETHOLOGY: Ethology = Ethology {
        sleep: glide_asleep,
        ..BLOOD_FEEDER
    };
}
