#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod aerogigantiform;
pub mod nubidriftonidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon =
    Taxon::new(&super::TAXON, "Caeliformes", "").ecology("herbivore", 0, false);
