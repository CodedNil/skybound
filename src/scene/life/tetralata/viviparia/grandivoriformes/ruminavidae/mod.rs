#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod herbivolus;
pub mod umbraevolidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ruminavidae", "").ecology("", 0, false);
