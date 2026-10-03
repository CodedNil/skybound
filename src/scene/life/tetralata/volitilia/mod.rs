#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod parvifuga;
pub mod petrolata;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Volitilia", "").ecology("", 0, false);
