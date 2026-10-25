use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Feeding},
    eyes, feathers, mouth, prepare_pose, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail, tapered_tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 2200.0,
    lifespan_days: 50000.0,
    feeding: Feeding::Scavenger,
    speed: 32.0,
    adult_mass: 20.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 49.8;

taxon!(
    Species,
    "Umbravola tacita",
    "Shadowings",
    [
        "Shadowings circle through broad currents on wings clothed in muted, light-absorbing plumage.",
        "Fine sensitivity to wind and thermal changes helps them find the disturbed air around a falling carcass.",
        "Mouth tentacles pull free tissue while the wing pairs match the remains' descent.",
        "Sharp, echoing cries call companions to food and warn other animals when a large predator remains nearby.",
        "Remembered routes and the always-visible polar beams guide their searches between meals that never remain in one place for long.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 8.0, 14.0, 0);
    c.opaque_bounds = vec3(7.0, 6.0, 21.0).extend(0.0);
    rig_tentacles(c, mouth(7.6, 1.15), 0.9, BIOLOGY.oral_pairs);
}

/// A dark, silent night-grazer: a lean, deep-keeled body built for long
/// glides, a narrow head with large eyes for the dim phase, and a single
/// swept horn over the brow.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(2.9, 3.1, 7.6), 1.15);
    let keel = rounded(p - vec3(0.0, -2.2, -1.0), vec3(1.0, 1.6, 4.5));
    let mut hit = Surface::new(smooth_min(smooth_min(body, keel, 1.0), tail(p, c), 0.45), 0);
    hit.join(feathers(p, c, 4.6), 1);
    hit.join(
        tapered_tail(p, vec3(0.0, 1.6, -8.4), vec3(0.0, 3.4, -5.4), 0.55, 0.06),
        7,
    );
    hit.join(eyes(p, 7.6, 1.15), 3);
    // A night grazer: wide downward eyes take in what little light the dim
    // aur gives.
    hit.join(
        rounded(
            vec3(p.x.abs(), p.y, p.z) - vec3(0.8, -0.8, -8.3),
            vec3(0.5, 0.42, 0.55),
        ),
        3,
    );
    hit.join(
        oral_tentacles(p, c, mouth(7.6, 1.15), 0.9, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.06, 0.08, 0.14, 0.0),
    vec3(0.16, 0.23, 0.29),
    [vec4(20.0, 5.0, -2.0, 0.98), vec4(13.0, 3.0, 2.8, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
