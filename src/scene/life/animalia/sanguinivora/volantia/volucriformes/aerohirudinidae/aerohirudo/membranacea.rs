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
    maturity_days: 60.0,
    lifespan_days: 1800.0,
    speed: 18.0,
    adult_mass: 0.06,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 24.2;

taxon!(
    Species,
    "Aerohirudo membranacea",
    "Air Leeches",
    [
        "Air Leeches hover beside busy feeding routes on broad translucent wings supported by fine visible struts.",
        "A narrow, flexible body hangs behind the feeding sucker, keeping the mouth steady as a passing host moves through the current.",
        "They take a brief meal of blood or sap and release before the host can turn toward the disturbance.",
        "Broad wings provide control in the sheltered air around plants but make exposed River currents difficult to cross.",
        "During the dim aur phase, crowded drinking sites offer better feeding opportunities than searching the dark open Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.5, 1.4, 5.5).extend(0.0);
}

/// A long, ringed flying leech, flattened beneath its membranes, with a
/// flared sucking cup for a mouth and rows of eyespots behind it.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let sway = (c.animation.z * 1.2 + p.z * 0.6).sin() * 0.04 * (p.z + 4.5).max(0.0);
    let q = p - vec3(sway, 0.0, 0.0);
    let mut hit = Surface::new(
        annulated(
            q,
            vec3(0.0, 0.0, -4.4),
            vec3(0.0, 0.1, 4.6),
            vec3(0.42, 0.75, 0.28),
            24.0,
            0.06,
            0.72,
        ),
        0,
    );
    hit.join(
        sucker(
            q,
            vec3(0.0, -0.05, -4.55),
            vec3(0.0, -0.3, -1.0).normalize(),
            0.58,
        ),
        8,
    );
    hit.join(eyespots(q, vec3(0.2, 0.24, -4.1), 0.3, 3, 0.055), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.29, 0.13, 0.19, 0.0),
    vec3(0.75, 0.27, 0.33),
    [vec4(8.0, 3.2, -1.2, 0.25), Vec4::ZERO],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
