use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, feathers, mouth, prepare_pose, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail, tapered_tail, torso,
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
    rig_tentacles(c, mouth(10.0, 1.6), 2.3, BIOLOGY.oral_pairs);
}

/// A big, long-necked grazer: a deep chest under the wing roots, a
/// humped back, a neck reaching forward to a long head, and a broad soft
/// lip beneath its feeding tentacles.
/// A big grazer built for long hauls between algae streams: a deep body
/// to carry its gut, a swollen gas store along the back to cut the cost of
/// carrying it, a broad cowl that sweeps drifting algae back to the mouth
/// as it flies, and a soft lip hung beneath for cropping floating mats.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(3.6, 3.3, 10.0), 1.6);
    let store = rounded(p - vec3(0.0, 2.3, -0.5), vec3(2.5, 1.5, 4.5));
    let mut hit = Surface::new(smooth_min(smooth_min(body, store, 1.4), tail(p, c), 0.5), 0);
    hit.join(feathers(p, c, 4.0), 1);
    let lip = mouth(10.0, 1.6);
    hit.join(
        tapered_tail(p, lip, lip + vec3(0.0, -1.4, -2.2), 0.7, 0.3),
        8,
    );
    hit.join(eyes(p, 10.0, 1.6), 3);
    hit.join(oral_tentacles(p, c, lip, 2.3, BIOLOGY.oral_pairs), 8);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.68, 0.54, 0.31, 0.0),
    vec3(0.87, 0.75, 0.54),
    [vec4(20.0, 6.0, -2.5, 0.98), vec4(14.0, 4.0, 3.5, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
