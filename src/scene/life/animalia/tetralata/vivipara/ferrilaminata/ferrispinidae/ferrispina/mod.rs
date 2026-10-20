pub use super::render_wings;
use crate::scene::life::{biology::Biology, tapered_tail};
use spirv_std::glam::{Vec3, vec3};
pub mod custos;
pub mod falcata;

pub const BIOLOGY: Biology = Biology {
    speed: 65.0,
    adult_mass: 25.0,
    ..super::BIOLOGY
};

taxon!(
    Genus,
    "Ferrispina",
    "Iron Spine",
    [
        "Members of Ferrispina are serpentine predators whose spinal blades deliver sweeping aerial attacks.",
        "Independent wing pairs turn the long body through tight arcs without requiring contact with a surface.",
        "Juvenile groups use coordinated approaches to separate prey, while mature animals can overpower larger targets alone.",
        "One branch has specialised in guarding nursery currents, exchanging wide-ranging pursuit for repeated defensive patrols.",
        "Their shared flexible spine and cutting pass distinguish these roles from the straight piercing attack of the snout-armed family.",
    ],
);

fn blade(p: Vec3, root: Vec3, tip: Vec3, width: f32) -> f32 {
    tapered_tail(vec3(p.x * 8.0, p.y, p.z), root, tip, width, 0.008) / 8.0
}
