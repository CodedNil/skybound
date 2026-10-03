#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::Taxon;
pub mod caeliformes;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon::new(&super::TAXON, "Bullidorsata", "Gentle balloon-like drifters buoyed by gas bladders. Sticky jelly skin gathers algae and plants.")
    .ecology("herbivore", 0, false);
