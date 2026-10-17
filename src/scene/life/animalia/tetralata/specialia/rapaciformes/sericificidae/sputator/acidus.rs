use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 75.0,
    lifespan_days: 1500.0,
    speed: 10.0,
    social: Social::Solitary,
    adult_mass: 0.12,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, capsule, ellipsoid, eyes};
use crate::scene::surface::{Ray, Surface, TraceResult};

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
    c.opaque_bounds = vec3(5.0, 5.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.7, 1.4, 2.8)), 5);
    hit.join(ellipsoid(p - vec3(0.0, 0.15, -2.2), vec3(1.2, 1.0, 1.2)), 5);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -3.20), 0.85, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(
        capsule(p, vec3(0.0, -0.3, -2.8), vec3(0.0, -0.2, -4.0), 0.3),
        8,
    );
    hit.join(eyes(p, 2.8, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.18, 0.44, 0.29, 0.0),
        vec3(0.28, 0.58, 0.62),
        [vec4(6.0, 2.2, 0.0, 0.95), vec4(4.0, 1.8, 3.0, 0.32)],
    )
}
