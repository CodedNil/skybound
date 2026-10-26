use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    eyespots, smooth_min, sucker,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::sanguinivora::volantia::WING_MUSCLE,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.11,
    maturity_days: 60.0,
    lifespan_days: 1800.0,
    adult_mass: 0.06,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 24.2;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Aerohirudo membranacea",
    "Air Leeches",
    [
        "Air Leeches hover beside busy feeding routes on broad translucent wings supported by fine visible struts.",
        "A narrow, flexible body hangs behind the feeding sucker, keeping the mouth steady as a passing host moves through the current.",
        "They take a brief meal of blood or sap and release before the host can turn toward the disturbance.",
        "Broad wings provide control in the sheltered air around plants but make exposed River currents difficult to cross.",
        "In the dim equatorial skies, crowded drinking sites offer better feeding opportunities than searching the dark open Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.6, 2.2, 5.8).extend(0.0);
}

/// A hoverer led by its mouth.
///
/// The sucker is a broad cup out in front, held level by the wings while a long, narrow, ringed
/// body hangs behind and below it like a pendulum, keeping the cup steady as a passing host moves
/// through the current. A row of eyespots arcs over the cup's rim, and the broad wings, folds of
/// the body wall like all its kin's, are stretched over many fine struts.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let sway = (c.animation.z * 1.0 + p.z * 0.5).sin() * 0.08 * (p.z + 2.6).max(0.0);
    let droop = 0.012 * (p.z + 2.6).max(0.0).powi(2);
    let q = p - vec3(sway, -droop, 0.0);
    let neck = annulated(
        p,
        vec3(0.0, 0.0, -4.0),
        vec3(0.0, -0.05, -2.4),
        vec3(0.5, 0.58, 0.48),
        5.0,
        0.06,
        0.9,
    );
    let tail = annulated(
        q,
        vec3(0.0, -0.05, -2.5),
        vec3(0.0, -0.1, 4.8),
        vec3(0.48, 0.42, 0.08),
        28.0,
        0.06,
        0.85,
    );
    let mut hit = Surface::new(smooth_min(neck, tail, 0.3), 0);
    hit.join(
        sucker(
            p,
            vec3(0.0, 0.0, -4.25),
            vec3(0.0, -0.1, -1.0).normalize(),
            0.95,
        ),
        8,
    );
    hit.join(eyespots(p, vec3(0.45, 0.55, -4.0), 0.22, 4, 0.06), 3);
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.04, -3.19), vec3(0.10, 0.10, 0.29)),
    site(PUMP, vec3(0.00, -0.04, -2.16), vec3(0.12, 0.12, 0.25)),
    site(CROP, vec3(0.00, -0.04, -0.31), vec3(0.17, 0.17, 0.17)),
    pair(WING_MUSCLE, vec3(0.18, 0.08, -1.44), vec3(0.07, 0.07, 0.07)),
    site(BROOD, vec3(0.00, -0.19, 1.13), vec3(0.10, 0.10, 0.10)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.29, 0.13, 0.19, 0.0),
    vec3(0.75, 0.27, 0.33),
    [vec4(8.5, 3.6, -2.8, 0.25), Vec4::ZERO],
)
.hinged([vec2(0.5, 0.2), Vec2::ZERO])
.styled([vec4(0.6, 0.95, 6.0, 0.0), vec4(0.6, 0.95, 6.0, 0.0)])
.framed([0.25, 0.25])
.skinned(vec4(5.0, 1.0, 1.2, -0.3))
.flying(vec4(2.00, 0.40, 0.55, 1.00))
.marked(vec4(3.00, 5.00, 0.80, 0.40), vec3(0.95, 0.80, 0.85));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        speed: 18.0,
        ..super::super::PHYSIOLOGY
    };
}
