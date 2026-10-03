#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod aerohirudo;
pub mod ptilohirudo;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Voluceridae", "").ecology("", 0, false);
