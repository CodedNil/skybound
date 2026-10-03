#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod lucentiae;
pub mod manducidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Detritivora", "").ecology("", 0, false);
