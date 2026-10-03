#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod ferrilaminata;
pub mod grandivoriformes;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Viviparia", "Live bearers including highly intelligent animals communicating through clicks, whistles and song.")
    .ecology("", 0, false);
