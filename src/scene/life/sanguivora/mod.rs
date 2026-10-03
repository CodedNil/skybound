#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod erratia;
pub mod voluceria;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Sanguivora", "").ecology("", 0, false);
