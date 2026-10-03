#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod ruminavidae;
pub mod vocanathidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Grandivoriformes", "").ecology("", 0, false);
