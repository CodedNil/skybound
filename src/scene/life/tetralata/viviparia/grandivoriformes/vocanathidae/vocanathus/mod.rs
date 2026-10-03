#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod antiquus;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Vocanathus", "").ecology("", 0, false);
