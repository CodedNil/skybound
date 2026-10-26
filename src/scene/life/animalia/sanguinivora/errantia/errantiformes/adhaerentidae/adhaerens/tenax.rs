use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    eyespots, sucker,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::GANGLION,
    animalia::physiology::MUSCLE,
    animalia::physiology::PUMP,
    organs::{Viscera, pair, site},
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

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

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

/// A flattened, ringed leech, narrow at the head and broad behind, gripping with a small sucker
/// around its mouth and a broad disc at its tail.
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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.03, -2.14), vec3(0.06, 0.09, 0.14)),
    site(PUMP, vec3(0.00, -0.05, -1.01), vec3(0.07, 0.10, 0.24)),
    site(CROP, vec3(0.00, -0.01, 0.71), vec3(0.18, 0.19, 0.19)),
    pair(MUSCLE, vec3(0.16, 0.10, -0.38), vec3(0.07, 0.10, 0.17)),
    site(BROOD, vec3(0.00, -0.05, 2.14), vec3(0.08, 0.10, 0.10)),
]);

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
