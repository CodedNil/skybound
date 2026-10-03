#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod aerogigantidae;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon =
    Taxon::new(&super::TAXON, "Aerogigantiform", "").ecology("herbivore", 0, false);
