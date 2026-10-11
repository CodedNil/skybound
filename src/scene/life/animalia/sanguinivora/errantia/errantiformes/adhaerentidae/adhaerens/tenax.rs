use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.03,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 20.0;

pub const TAXON: Taxon = Taxon {
    name: "Adhaerens tenax",
    english_name: "Attached Ones",
    description: "Attached Ones begin as windborne leeches and settle permanently when they encounter an animal. \
        Their flattened, segmented bodies fit close to the host, and their exterior develops colours and patterns resembling its fur. \
        A broad sucker maintains contact while they draw fluids slowly. \
        Air currents carry unattached juveniles between potential hosts.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

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
