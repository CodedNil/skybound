#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod dissimuliformes;
pub mod voluceriformes;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Voluceria", "").ecology("", 0, false);
