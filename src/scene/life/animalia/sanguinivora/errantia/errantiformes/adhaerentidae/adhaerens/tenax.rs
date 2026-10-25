use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    eyespots, sucker,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 15.0,
    lifespan_days: 900.0,
    adult_mass: 0.03,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 20.0;

taxon!(
    Species,
    "Adhaerens tenax",
    "Attached Ones",
    [
        "Attached Ones begin as flattened, segmented juveniles carried between animals by the updraft.",
        "A broad sucker secures the first suitable host and becomes the centre of a permanent feeding attachment.",
        "As the body grows, its colours and surface texture resemble the host's fur, concealing the parasite from casual grooming.",
        "Small, sustained meals limit the disturbance that would provoke removal or attract Sweepers.",
        "Dense herds and shared drinking sites offer the young better chances of attachment than the sparse air between feeding communities.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.3, 1.1, 4.2).extend(0.0);
}

/// A flattened, ringed leech, narrow at the head and broad behind, gripping
/// with a small sucker around its mouth and a broad disc at its tail.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let sway = (c.animation.z * 1.5 + p.z * 0.9).sin() * 0.05 * (p.z + 3.2).max(0.0);
    let q = p - vec3(sway, 0.0, 0.0);
    let mut hit = Surface::new(
        annulated(
            q,
            vec3(0.0, 0.0, -3.0),
            vec3(0.0, -0.05, 2.9),
            vec3(0.32, 0.9, 0.55),
            17.0,
            0.07,
            0.48,
        ),
        0,
    );
    hit.join(
        sucker(
            q,
            vec3(0.0, -0.1, -3.05),
            vec3(0.0, -0.45, -0.9).normalize(),
            0.36,
        ),
        8,
    );
    hit.join(
        sucker(
            q,
            vec3(0.0, -0.3, 3.0),
            vec3(0.0, -0.9, 0.45).normalize(),
            0.7,
        ),
        4,
    );
    hit.join(eyespots(q, vec3(0.12, 0.13, -2.8), 0.2, 3, 0.04), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.39, 0.29, 0.21, 0.0),
    vec3(0.69, 0.54, 0.32),
    [Vec4::ZERO; 2],
)
.skinned(vec4(3.0, 1.0, 2.0, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
