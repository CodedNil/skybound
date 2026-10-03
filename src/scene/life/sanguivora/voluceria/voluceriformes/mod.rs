#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod voluceridae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Voluceriformes", "").ecology("", 0, false);
