use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::biology::Biology;
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid, eyes, feathers, prepare_pose};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 21.0,
    lifespan_days: 480.0,
    speed: 24.0,
    adult_mass: 0.01,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 30.4;

taxon!(
    Species,
    "Ptilohirudo pennata",
    "Feathered Leeches",
    [
        "Feathered Leeches glide along cloud margins on finely fringed wings surrounding a compact body.",
        "Their vision detects weak aur-lit contrasts, helping them find hosts hidden by mist.",
        "A quick attachment supplies a fluid meal before they release and return to cover.",
        "They favour broken cloud edges where passing animals are visible without requiring a long flight through dry air.",
        "They favour humid cloud pockets on the warmer margins of the equatorial Highlands, avoiding both frozen Shard Peaks and the strongest poleward winds.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 5.0, 0);
    c.opaque_bounds = vec3(4.0, 2.0, 10.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 0.55, 2.8)), 0);
    hit.join(feathers(p, c, 2.0), 1);
    hit.join(eyes(p, 2.8, 0.5), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.25, 0.32, 0.42, 0.0),
        vec3(0.69, 0.73, 0.80),
        [vec4(9.0, 3.0, 0.0, 0.98)],
    )
}
