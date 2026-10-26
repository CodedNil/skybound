pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod lithosoma;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 80.0,
    altitude: (60.0, 1500.0),
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Family,
    "Lithosomatidae",
    "Stone Bodies",
    [
        "Members of this family are mineral-armoured herbivores adapted to the strongest stable currents of the lower Gardens.",
        "Broad lifting surfaces slow descent rather than supporting the agile flight of lighter tetralates.",
        "Dense protective tissues permit contact with tough root growth and drifting mineral debris.",
        "Their dependence on Root-Stars keeps them near the lower edge of inhabited air without granting tolerance of pure aur.",
        "Weak or shifting updraft makes access to a reliable rising column central to their survival.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{Kind, animalia::physiology::Physiology, organs::Function};
    pub const PHYSIOLOGY: Physiology = Physiology {
        crops: Function::Drawing,
        speed: 1.0,
        eats: |_, food| food == Kind::Stelloradix,
        ..super::super::PHYSIOLOGY
    };
}
