use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{curved_tube, oral_tentacles, rig_tentacles},
    biology::{Biology, Feeding, Social},
    eyes, mouth, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};
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
        "Glowgnomes are tiny luminous sap feeders that inhabit sheltered burrows in Whisperwood.",
        "They retain the single wing pair of their parasitic ancestors, but coordinated wing strokes now help draw the host toward favourable currents.",
        "Pheromone signals carried through six vertical sap veins recruit groups on different sides of the trunk.",
        "Their light draws visitors through nearby flowering growth during the dim aur phase, while sap and wood cavities provide food and shelter.",
        "Limited feeding and shared movement have turned an ancestral parasitic visit into a partnership on which the drifting tree depends.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 4.0).extend(0.0);
    rig_tentacles(c, mouth(1.35, 0.4), 0.35, BIOLOGY.oral_pairs);
}

/// A small, round-bodied flyer carrying its light ahead of it: a glowing
/// lure hung from a stalk arching up off the crown, like a lantern on a
/// pole, its glow cupped in a fleshy hood.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let sway = (c.animation.z * 1.1).sin() * 0.12;
    let body = torso(p, vec3(0.8, 0.85, 1.35), 0.4);
    let stalk = curved_tube(
        p,
        vec3(0.0, 0.75, -0.7),
        vec3(0.0, 2.5, -0.9),
        vec3(sway, 2.25, -2.3),
        0.11,
    );
    let mut hit = Surface::new(smooth_min(body, stalk, 0.2), 0);
    let lure = p - vec3(sway, 1.85, -2.45);
    let hood = rounded(lure - vec3(0.0, 0.28, 0.05), vec3(0.5, 0.36, 0.52));
    hit.join(hood.max(-rounded(lure, vec3(0.42, 0.48, 0.42))), 0);
    hit.join(rounded(lure, vec3(0.34, 0.42, 0.34)), 9);
    hit.join(
        oral_tentacles(p, c, mouth(1.35, 0.4), 0.35, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 1.35, 0.4), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.55, 0.32, 0.15, 0.0),
    vec3(1.0, 0.72, 0.20),
    [vec4(3.0, 1.3, -0.3, 0.32), Vec4::ZERO],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
