pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod aculeorhynchidae;
pub mod ferrispinidae;
pub mod naeturidae;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Order,
    "Ferrilaminata",
    "Iron Blades",
    [
        "Members of this order descend from live-bearing four-winged animals adapted to powerful, controlled flight.",
        "Predatory branches arm the head or the spine with iron for different aerial strikes, or armour the hide with it against the hunters that come for a still ambusher.",
        "Their shared internal brooding and prolonged juvenile development also favour recognition and care of companions.",
        "In the Næ lineage, ancestral pursuit and handling abilities have shifted toward manipulation, cooperation and specialised aur protection.",
        "The order therefore retains a common flight and reproductive anatomy despite large differences in diet and adult size.",
        "Every member eats metal. The order arose in the mineral haze, and its bodies still take iron from the dust they breathe, the stony hides they scrape and the graves they pick, and lay it down wherever their lineage needs hardness: blades and plates, a stylet, or the sheaths of aur sacs. None is born with it, so the old carry far more than the young.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Physiology, Take, hunts};

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: hunts,
        takes: Take::Prey,
        speed: 45.0,
        ironclad: true,
        ..super::super::PHYSIOLOGY
    };
}
