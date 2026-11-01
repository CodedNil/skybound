pub use super::render_wings;
use crate::scene::life::biology::Biology;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::tetralata::SPRING_ROD,
    organs::{Function, Organ, Part, Tissue},
};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;
pub mod aculeorhynchidae;
pub mod ferrispinidae;
pub mod naeturidae;

pub const BIOLOGY: Biology = super::BIOLOGY;

/// The spring rods of the metal-eaters, their fibre laced with iron laid down over a life: stiffer
/// and heavier, giving back a harder stroke, and turning blows that would cut a plain rod.
#[cfg(not(target_arch = "spirv"))]
pub const IRON_ROD: Organ =
    Organ::new("iron-laced rod", Function::Support, vec3(0.44, 0.44, 0.47)).of(Tissue::Ironbound);

/// The live-bearers' frame with its rods laced with iron, thicker for the load.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[Part::rod(
    IRON_ROD,
    vec3(-0.85, 0.6, 0.22),
    vec3(0.95, 0.5, 0.22),
    0.075,
)
.replacing(SPRING_ROD)];

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
        "Every member eats metal. The order arose in the mineral haze, and its bodies still take iron from the dust they breathe, the stony hides they scrape and the graves they pick, and lay it down wherever their lineage needs hardness: blades and plates, a stylet, or the sheaths of aur sacs, and in every one of them the spring rods of its frame. None is born with it, so the old carry far more than the young.",
    ],
    anatomy: ANATOMY,
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
