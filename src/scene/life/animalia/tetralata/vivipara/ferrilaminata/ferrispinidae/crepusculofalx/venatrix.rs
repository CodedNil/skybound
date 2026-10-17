use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 730.0,
    lifespan_days: 14600.0,
    speed: 45.0,
    social: Social::Solitary,
    adult_mass: 5.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{
    CreatureInstance, ellipsoid, eyes, feathers, prepare_pose, smooth_min, tail, torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Crepusculofalx venatrix",
    "Duskslicers",
    [
        "Duskslicers conceal muted bodies in the mineral haze while displaying a small luminous lure on the feathered tail.",
        "Small animals approaching the apparent feeding signal enter the sweep of sharpened wing margins.",
        "A brief banking strike captures prey without exposing the whole body in a long chase.",
        "Mouth tentacles secure the meal, and scavenging provides food when the lure attracts little attention.",
        "Their dim-field specialisation makes clear, brightly lit Gardens less favourable than the mineral-rich currents that many other animals only visit.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 5.0, 14.0, 0);
    c.opaque_bounds = vec3(6.0, 5.0, 21.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.6, 1.4, 5.2), 0.8), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    hit.join(feathers(p, c, 3.0), 1);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -5.60), 0.80, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(
        ellipsoid(p - c.tail[5].truncate(), vec3(0.35, 0.35, 0.65)),
        9,
    );
    hit.join(eyes(p, 5.2, 0.8), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.22, 0.30, 0.20, 0.0),
        vec3(0.79, 0.43, 0.24),
        [vec4(13.0, 3.0, 0.0, 0.98), vec4(8.0, 2.0, 6.0, 0.98)],
    )
}
