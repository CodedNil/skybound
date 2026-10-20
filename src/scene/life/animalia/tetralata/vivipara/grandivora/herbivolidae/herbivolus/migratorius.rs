use crate::scene::life::{
    CreatureInstance,
    animalia::tetralata::oral_tentacles,
    biology::Biology,
    capsule, eyes, feathers, prepare_pose, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1460.0,
    lifespan_days: 40000.0,
    speed: 12.0,
    adult_mass: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 49.8;

taxon!(
    Species,
    "Herbivolus migratorius",
    "Grazers",
    [
        "Grazers are large herbivores with broad feathered wings and soft, tongue-like mouth tentacles.",
        "They spread those appendages through dense algae concentrations, gathering food while the wing pairs hold a steady course.",
        "Herds space themselves across a rising food stream so that neighbours do not continually feed in one another's depleted wake.",
        "Repeated journeys to the Mineral Fields replenish minerals absent from clearer feeding air.",
        "Their migration routes link productive Gardens to dusty regions, carrying predators, cleaners and attached parasites between the same communities.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 8.0, 14.0, 0);
    c.opaque_bounds = vec3(7.0, 6.0, 22.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(4.8, 4.2, 9.0), 1.4), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.5);
    hit.join(feathers(p, c, 4.0), 1);
    hit.join(
        capsule(p, vec3(0.0, -1.0, -11.0), vec3(0.0, -3.0, -17.0), 0.45),
        8,
    );
    hit.join(eyes(p, 9.0, 1.4), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -13.0), 2.3, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.68, 0.54, 0.31, 0.0),
        vec3(0.87, 0.75, 0.54),
        [vec4(20.0, 6.0, 0.0, 0.98), vec4(14.0, 4.0, 10.0, 0.98)],
    )
}
