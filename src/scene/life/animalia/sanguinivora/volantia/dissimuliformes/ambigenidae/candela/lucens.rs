use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::{Biology, Feeding, Social},
    capsule, eyespots, rounded, smooth_min, sucker,
    surface::{Ray, Surface, TraceResult},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 12.0,
    lifespan_days: 240.0,
    feeding: Feeding::Grazer,
    social: Social::Mutualist,
    adult_mass: 0.002,
    speed: 1.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 16.0;

taxon!(
    Species,
    "Candela lucens",
    "Glowgnomes",
    [
        "Glowgnomes are tiny luminous sap feeders that inhabit sheltered burrows in Whisperwood, their soft ringed fronts trailing a swollen, translucent lamp sac hooped with ribs and laced with the sap veins that feed its light.",
        "They retain the single wing pair of their parasitic ancestors, but coordinated wing strokes now help draw the host toward favourable currents.",
        "Pheromone signals carried through six vertical sap veins recruit groups on different sides of the trunk.",
        "Their light draws visitors through nearby flowering growth during the dim aur phase, while sap and wood cavities provide food and shelter.",
        "Limited feeding and shared movement have turned an ancestral parasitic visit into a partnership on which the drifting tree depends.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(2.5, 2.0, 2.6).extend(0.0);
}

/// A burrow-dweller that is mostly lamp.
///
/// Its front is a short, soft, ringed grub with a small sucker and a fringe for
/// sipping sap, and only faint eyespots, for it lives in the dark of the wood;
/// its whole rear has swollen into a translucent sac lit from within, hooped by
/// ribs and laced with the sap veins that feed the light. Stubby wings are
/// enough to carry it between burrows and to beat in time with its neighbours,
/// drawing the tree towards better air.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let grub = annulated(
        p,
        vec3(0.0, 0.0, -1.7),
        vec3(0.0, 0.05, 0.1),
        vec3(0.32, 0.5, 0.42),
        5.0,
        0.1,
        0.92,
    );
    let lamp = rounded(p - vec3(0.0, 0.1, 1.0), vec3(0.78, 0.72, 1.0));
    let mut hit = Surface::new(smooth_min(grub, lamp + 0.03, 0.2), 0);
    hit.join(lamp, 9);
    // Ribs hooping the lamp, darker than the light between them.
    let q = p - vec3(0.0, 0.1, 1.0);
    for rib in 0..4 {
        let z = rib as f32 * 0.45 - 0.68;
        let across = (1.0 - (z / 1.0).powi(2)).max(0.05).sqrt();
        let hoop = vec2(
            vec2(q.x / 0.78, q.y / 0.72).length() * across * 0.75 - across * 0.75,
            q.z - z,
        )
        .length()
            - 0.04;
        hit.join(hoop, 0);
    }
    let facing = vec3(0.0, -0.3, -1.0).normalize();
    hit.join(sucker(p, vec3(0.0, -0.05, -1.75), facing, 0.28), 8);
    let mouth = p - vec3(0.0, -0.05, -1.8);
    let around = mouth.y.atan2(mouth.x);
    let sector = TAU / 8.0;
    let turn = (around / sector).round() * sector;
    let (sine, cosine) = turn.sin_cos();
    let sway = (c.animation.z * 1.5 + turn).sin() * 0.04;
    hit.join(
        capsule(
            mouth,
            vec3(cosine * 0.25, sine * 0.25, 0.0),
            vec3(cosine * (0.38 + sway), sine * (0.38 + sway), -0.22),
            0.025,
        ),
        8,
    );
    hit.join(eyespots(p, vec3(0.12, 0.3, -1.55), 0.12, 2, 0.025), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.55, 0.32, 0.15, 0.0),
    vec3(1.0, 0.72, 0.20),
    [vec4(2.0, 1.1, -0.6, 0.35), Vec4::ZERO],
)
.hinged([vec2(0.42, 0.3), Vec2::ZERO])
.styled([vec4(0.3, 1.0, 0.0, 0.0), vec4(0.3, 1.0, 0.0, 0.0)])
.framed([0.4, 0.4])
.skinned(vec4(5.0, 0.8, 1.6, -0.2))
.flying(vec4(4.00, 0.40, 0.20, 0.60))
.marked(vec4(5.00, 4.00, 0.70, 0.40), vec3(1.00, 0.85, 0.40));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
