use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    capsule, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
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
    maturity_days: 1.0,
    lifespan_days: 90.0,
    adult_mass: 0.001,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 17.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Haematozoon invasor",
    "Blood Dwellers",
    [
        "Blood Dwellers spend most of their lives dividing within the fluid circulation of a host.",
        "Their small, soft bodies obtain nutrients directly from that enclosed environment rather than maintaining an exposed feeding grip.",
        "Dispersal growths develop near the host's surface and eventually release compact juveniles into the air.",
        "These juveniles have only a brief opportunity to enter another host before their reserves are exhausted.",
        "Crowded feeding grounds and close contact between animals therefore shape their spread more strongly than the long-distance wind that carries them.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.0, 1.0, 3.1).extend(0.0);
}

/// A ringed worm swollen with blood, daughters budding from one flank and growing larger towards
/// the tail, its proboscis armed with hooks.
pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let body = annulated(
        p,
        vec3(0.0, 0.0, -2.2),
        vec3(0.0, 0.05, 2.3),
        vec3(0.28, 0.5, 0.2),
        12.0,
        0.08,
        0.88,
    );
    let mut buds = 1e6f32;
    for i in 0..4 {
        let f = i as f32;
        let at = vec3(0.34 + f * 0.04, 0.16 - f * 0.03, -0.7 + f * 0.6);
        buds = buds.min(rounded(p - at, vec3(0.9, 0.85, 1.1) * (0.14 + f * 0.05)));
    }
    let mut hit = Surface::new(
        smooth_min(body, buds, 0.12),
        if buds < body { 2 } else { 0 },
    );
    hit.join(
        tapered_tail(p, vec3(0.0, 0.0, -2.1), vec3(0.0, -0.04, -2.8), 0.2, 0.07),
        8,
    );
    for i in 0..5 {
        let angle = i as f32 * 1.256_6;
        let around = vec3(angle.cos(), angle.sin(), 0.0);
        let root = vec3(0.0, -0.03, -2.6) + around * 0.1;
        hit.join(
            capsule(p, root, root + around * 0.13 + vec3(0.0, 0.0, 0.1), 0.025),
            7,
        );
    }
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.03, -1.63), vec3(0.06, 0.06, 0.18)),
    site(PUMP, vec3(0.00, -0.03, -0.75), vec3(0.07, 0.07, 0.24)),
    site(CROP, vec3(0.00, 0.01, 0.53), vec3(0.18, 0.18, 0.21)),
    pair(MUSCLE, vec3(0.16, 0.09, -0.17), vec3(0.07, 0.07, 0.17)),
    site(BROOD, vec3(0.00, -0.03, 1.63), vec3(0.08, 0.08, 0.09)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.54, 0.13, 0.22, 0.0),
    vec3(0.87, 0.43, 0.46),
    [Vec4::ZERO; 2],
)
.skinned(vec4(5.0, 1.0, 1.4, -0.30));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        mates: false,
        ..super::super::PHYSIOLOGY
    };
}
