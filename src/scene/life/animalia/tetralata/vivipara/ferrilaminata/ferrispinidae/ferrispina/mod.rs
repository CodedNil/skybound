use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 65.0,
    adult_mass: 25.0,
    ..super::BIOLOGY
};
pub use super::render_wings;
use crate::scene::life::tapered_tail;
use crate::scene::life::{Rank, Taxon};
use spirv_std::glam::{Vec3, vec3};

pub mod custos;
pub mod falcata;

fn blade(p: Vec3, root: Vec3, tip: Vec3, width: f32) -> f32 {
    tapered_tail(vec3(p.x * 8.0, p.y, p.z), root, tip, width, 0.008) / 8.0
}

pub const TAXON: Taxon = Taxon {
    name: "Ferrispina",
    english_name: "Iron Spine",
    description: "Shredders are agile, serpentine apex predators with razor-sharp blades along their spines.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};
