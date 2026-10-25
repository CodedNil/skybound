use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Social},
    eyes, mouth, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 75.0,
    lifespan_days: 1500.0,
    speed: 10.0,
    social: Social::Solitary,
    adult_mass: 0.12,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 24.7;

taxon!(
    Species,
    "Sputator acidus",
    "Spitfire Beetles",
    [
        "Spitfire Beetles are compact, armoured predators that project the adhesive secretion inherited from silk-making ancestors.",
        "In this branch the secretion remains a sticky acidic droplet, arresting small grazers on a frond or suspended strand.",
        "Mouth tentacles grip vegetation while the wings brace against recoil and the movement of the target.",
        "A short attack across a feeding passage avoids the energy cost of chasing prey into open air.",
        "They often hunt beside Weaver corridors, where concentrated traffic rewards precision but careless discharge can destroy a useful anchor.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 5.0, 5.2).extend(0.0);
    rig_tentacles(c, mouth(2.4, 0.7), 0.85, BIOLOGY.oral_pairs);
}

/// A spitter, armoured against its own acid: overlapping plates across a
/// broad back, and a muscular siphon thrust out beneath the face.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut body = torso(p, vec3(1.5, 1.25, 2.4), 0.7);
    for i in 0..5 {
        let z = i as f32 * 1.0 - 2.0;
        let plate = rounded(
            p - vec3(0.0, 0.85 - (z * 0.35).powi(2), z),
            vec3(1.45 - (z * 0.2).abs(), 0.42, 0.62),
        );
        body = smooth_min(body, plate, 0.08);
    }
    let mut hit = Surface::new(body, 5);
    hit.join(
        tapered_tail(
            p,
            vec3(0.0, -0.35, -4.0),
            vec3(0.0, -0.45, -4.95),
            0.3,
            0.14,
        ),
        8,
    );
    hit.join(
        oral_tentacles(p, c, mouth(2.4, 0.7), 0.85, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 2.4, 0.7), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.18, 0.44, 0.29, 0.0),
    vec3(0.28, 0.58, 0.62),
    [vec4(6.0, 2.2, -0.6, 0.95), vec4(4.0, 1.8, 1.0, 0.32)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
