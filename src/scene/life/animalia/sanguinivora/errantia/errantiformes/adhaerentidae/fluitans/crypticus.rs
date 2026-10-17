use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 18.0,
    lifespan_days: 600.0,
    adult_mass: 0.02,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, ellipsoid, prepare_pose, ribbons};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 26.0;

taxon!(
    Species,
    "Fluitans crypticus",
    "Floating Ones",
    [
        "Floating Ones begin life among kelp, where green-brown bodies and swaying lateral folds blend with the fronds.",
        "Instead of searching open air, they wait in the narrow spaces through which feeding animals draw vegetation.",
        "A passing grazer brings the sucker within reach, allowing the parasite to leave the plant and settle permanently on the host.",
        "The trailing folds then lie against the new surface, reducing drag during travel.",
        "Dense Jungles favour this ambush life cycle, while the movement of kelp mats carries unattached young into new feeding routes.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 9.0, 2);
    c.opaque_bounds = vec3(3.0, 3.0, 18.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 0.55, 4.0)), 0);
    hit.join(ribbons(p, c, 2), 4);
    for i in 0..6 {
        let z = i as f32 * 1.2 - 3.0;
        hit.join(
            ellipsoid(vec3(p.x.abs() - 1.1, p.y, p.z - z), vec3(1.4, 0.12, 0.6)),
            4,
        );
    }
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.24, 0.39, 0.19, 0.0),
        vec3(0.53, 0.61, 0.25),
        [],
    )
}
