#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod nubiballona;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon =
    Taxon::new(&super::TAXON, "Leviformidae", "").ecology("herbivore", 0, false);
