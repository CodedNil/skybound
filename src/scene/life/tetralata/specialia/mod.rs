#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod detritivora;
pub mod rapaxina;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Specialia", "").ecology("", 0, false);
