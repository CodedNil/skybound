#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod vocanathus;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Vocanathidae", "").ecology("", 0, false);
