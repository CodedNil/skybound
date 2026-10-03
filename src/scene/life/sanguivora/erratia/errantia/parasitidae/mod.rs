#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod hematozoa;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Parasitidae", "").ecology("", 0, false);
