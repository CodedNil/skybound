#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod lucinani;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Candelis", "").ecology("", 0, false);
