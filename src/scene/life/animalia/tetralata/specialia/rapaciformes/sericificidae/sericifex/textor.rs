pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    annulated,
    biology::Biology,
    capsule, eyes, eyespots, mouth, smooth_min,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 90.0,
    lifespan_days: 2000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 36.7;

taxon!(
    Species,
    "Sericifex textor",
    "Weavers",
    [
        "Weavers hold themselves between kelp fronds on four wings while mouth tentacles draw out and fasten silk.",
        "Loose catching strands hang across feeding approaches, while stronger braided lines keep neighbouring plants from drifting beyond reach.",
        "They read changes in tension to distinguish grazing visitors from moving anchors, then reel captured prey toward the sheltered centre.",
        "Lines are loosened or replaced as currents shift, allowing the structure to breathe with a Jungle rather than resist it as a rigid net.",
        "Nests, collected bright fragments and the work of Næ crafters accumulate along old anchor corridors, turning a feeding apparatus into a persistent aerial neighbourhood.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 6.0).extend(0.0);
    rig_tentacles(c, mouth(1.6, 0.6), 0.65, BIOLOGY.oral_pairs);
}

/// A silk-spinner: a compact forebody with a spare pair of eyespots above
/// the true eyes, and a heavy, ringed abdomen ending in spinnerets.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let front = torso(p, vec3(1.1, 0.95, 1.6), 0.6);
    let abdomen = annulated(
        p,
        vec3(0.0, 0.15, 0.5),
        vec3(0.0, -0.25, 4.1),
        vec3(0.9, 1.35, 0.38),
        5.0,
        0.1,
        0.85,
    );
    let mut hit = Surface::new(smooth_min(front, abdomen, 0.35), 0);
    for i in 0..3 {
        let angle = (i as f32 - 1.0) * 0.6;
        let root = vec3(angle.sin() * 0.25, -0.35 - angle.cos() * 0.05, 4.15);
        hit.join(
            capsule(p, root, root + vec3(angle.sin() * 0.2, -0.15, 0.4), 0.09),
            7,
        );
    }
    hit.join(eyes(p, 1.6, 0.6), 3);
    hit.join(eyespots(p, vec3(0.45, 1.15, -1.8), 0.3, 2, 0.08), 3);
    hit.join(
        oral_tentacles(p, c, mouth(1.6, 0.6), 0.65, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.35, 0.24, 0.44, 0.0),
    vec3(0.83, 0.58, 0.30),
    [vec4(6.0, 2.0, -0.5, 0.27), vec4(4.0, 1.5, 0.6, 0.27)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
