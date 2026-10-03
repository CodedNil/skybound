#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod textricinae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Rapaxina", "").ecology("", 0, false);
