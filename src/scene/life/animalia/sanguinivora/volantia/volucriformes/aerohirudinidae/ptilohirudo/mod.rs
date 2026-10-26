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
    //! The Feathered Leeches live at the margins of cloud. They glide its broken edges, where a
    //! passing host shows but they stay hidden, sleep on the wing, and after a meal slip back into
    //! the vapour before the host can turn.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges, glide_asleep},
            sanguinivora::ETHOLOGY as BLOOD_FEEDER,
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        sleep: glide_asleep,
        ..BLOOD_FEEDER
    };

    /// Back into cover: up towards the wetter air of the cloud.
    const RETREAT: Intent = Intent {
        name: "retreat to cloud",
        act: |_, s| Goal::new(Activity::Return, Some(s.up() * 80.0), 0.8),
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (BLOOD_FEEDER.urges)(s, urges);
        // Fed, or caught out in the dry, it makes for cover.
        let exposed = (0.55 - s.seen.humidity).max(0.0) * 1.5;
        urges.add(RETREAT, exposed + (1.0 - s.drives.hunger) * 0.3);
    }
}
