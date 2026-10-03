#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod ramiphagae;
pub mod terrapodae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(
    &super::TAXON,
    "Petrolata",
    "Dense, armored broad-winged herbivores.",
)
.ecology("herbivore", 0, false);
