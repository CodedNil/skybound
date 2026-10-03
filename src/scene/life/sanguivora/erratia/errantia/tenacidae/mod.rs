#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod adherans;
pub mod fluitans;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Tenacidae", "").ecology("", 0, false);
