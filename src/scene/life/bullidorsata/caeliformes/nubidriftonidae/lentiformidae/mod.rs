#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod levita;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon =
    Taxon::new(&super::TAXON, "Lentiformidae", "").ecology("herbivore", 0, false);
