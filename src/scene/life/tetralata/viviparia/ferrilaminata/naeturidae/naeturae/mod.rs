#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod naetu;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Næturae", "").ecology("omnivore", 0, true);
