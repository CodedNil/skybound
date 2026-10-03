#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod luminiscidae;
pub mod nebulaphoridae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Parvifuga", "").ecology("", 0, false);
