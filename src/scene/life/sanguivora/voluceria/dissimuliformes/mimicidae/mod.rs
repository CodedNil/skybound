#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod ambigena;
pub mod candelis;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Mimicidae", "").ecology("", 0, false);
