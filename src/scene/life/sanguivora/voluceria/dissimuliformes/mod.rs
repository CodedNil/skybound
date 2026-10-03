#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod mimicidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Dissimuliformes", "").ecology("", 0, false);
