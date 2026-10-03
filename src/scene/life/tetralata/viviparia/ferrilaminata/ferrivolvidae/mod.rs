#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod crepusculorbis;
pub mod ferrispinae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ferrivolvidae", "").ecology("", 0, false);
