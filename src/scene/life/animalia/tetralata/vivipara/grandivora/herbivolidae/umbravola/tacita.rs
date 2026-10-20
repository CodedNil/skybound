use crate::scene::life::{
    CreatureInstance,
    animalia::tetralata::oral_tentacles,
    biology::{Biology, Feeding},
    eyes, feathers, prepare_pose, smooth_min,
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
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(3.4, 3.7, 7.2), 1.0), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.45);
    hit.join(feathers(p, c, 4.6), 1);
    hit.join(
        tapered_tail(p, vec3(0.0, 0.5, -9.0), vec3(0.0, -0.4, -12.0), 0.8, 0.1),
        6,
    );
    hit.join(eyes(p, 7.2, 1.0), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -10.0), 0.9, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.06, 0.08, 0.14, 0.0),
        vec3(0.16, 0.23, 0.29),
        [vec4(20.0, 5.0, 0.0, 0.98), vec4(13.0, 3.0, 9.0, 0.98)],
    )
}
