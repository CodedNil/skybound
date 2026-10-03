#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod naeturae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(
    &super::TAXON,
    "Næturidae",
    "The only lineage able to directly process aur.",
)
.ecology("omnivore", 0, true);
