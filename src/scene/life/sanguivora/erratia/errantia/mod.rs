#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod parasitidae;
pub mod tenacidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Errantia", "").ecology("", 0, false);
