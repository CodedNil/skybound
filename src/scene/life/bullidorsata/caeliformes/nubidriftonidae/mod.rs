#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod lentiformidae;
pub mod leviformidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon =
    Taxon::new(&super::TAXON, "Nubidriftonidae", "").ecology("herbivore", 0, false);
