pub use super::{BIOLOGY, render};

pub mod pomum;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Family,
    "Pomaceae",
    "Fruits",
    [
        "Members of this family form rounded buoyant fruits with thick, iridescent aur-collecting rinds.",
        "Nutrient-rich interiors attract grazers and other animals into loose fields of independently drifting plants.",
        "Seeds develop feathery surfaces, steering structures and attachment filaments before release.",
        "These features spread the risks of dispersal between open current and transport on a living carrier.",
        "The family retains vascular supply to its reproductive tissues while replacing large display petals with a durable, edible dispersal body.",
    ],
);
