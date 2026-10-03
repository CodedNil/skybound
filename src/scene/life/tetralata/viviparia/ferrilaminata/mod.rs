#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod aculeorhinidae;
pub mod ferrivolvidae;
pub mod naeturidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Ferrilaminata", "").ecology("", 0, false);
