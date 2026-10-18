use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, eyes, gills, prepare_pose, smooth_min, tail, torso};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 2200.0,
    lifespan_days: 42000.0,
    social: Social::Guardian,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Ferrispina custos",
    "Iron Spined Guardians",
    [
        "Iron Spined Guardians are heavy-bodied Shredders that remain near communal nursery currents.",
        "Conspicuous spinal blades and repeated banking patrols discourage predators from approaching slowly developing young.",
        "Their attacks retain the sweeping motion of roaming relatives, but begin from a defended area rather than a prolonged pursuit.",
        "Vocal exchanges and the motion of nearby adults help maintain a loose moving boundary around the nursery.",
        "The nursery travels with a suitable current, so protection depends on recognising companions rather than defending a fixed place.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 6.0, 14.0, 0);
    c.opaque_bounds = vec3(6.0, 7.0, 27.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.5, 2.3, 7.0), 1.1), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.5);
    for i in 0..7 {
        let z = i as f32 * 2.0 - 5.5;
        hit.join(
            super::blade(p, vec3(0.0, 1.8, z), vec3(0.0, 5.6, z + 2.0), 0.7),
            6,
        );
    }
    hit.join(gills(p, 7.0, 2), 9);
    hit.join(eyes(p, 7.0, 1.1), 3);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -10.0), 1.1, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.26, 0.45, 0.43, 0.0),
        vec3(0.86, 0.68, 0.28),
        [vec4(17.0, 3.0, 0.0, 0.98), vec4(11.0, 2.2, 7.0, 0.98)],
    )
}
