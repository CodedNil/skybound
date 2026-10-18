use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::biology::Biology;
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 15.0,
    lifespan_days: 900.0,
    adult_mass: 0.03,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 20.0;

taxon!(
    Species,
    "Adhaerens tenax",
    "Attached Ones",
    [
        "Attached Ones begin as flattened, segmented juveniles carried between animals by the updraft.",
        "A broad sucker secures the first suitable host and becomes the centre of a permanent feeding attachment.",
        "As the body grows, its colours and surface texture resemble the host's fur, concealing the parasite from casual grooming.",
        "Small, sustained meals limit the disturbance that would provoke removal or attract Sweepers.",
        "Dense herds and shared drinking sites offer the young better chances of attachment than the sparse air between feeding communities.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.2, 1.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.9, 0.5, 3.2)), 0);
    for i in 0..8 {
        let z = i as f32 * 0.7 - 2.5;
        hit.join(ellipsoid(p - vec3(0.0, 0.0, z), vec3(0.94, 0.53, 0.23)), 4);
    }
    hit.join(
        ellipsoid(p - vec3(0.0, -0.3, -3.0), vec3(0.7, 0.25, 0.6)),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.39, 0.29, 0.21, 0.0),
        vec3(0.69, 0.54, 0.32),
        [],
    )
}
