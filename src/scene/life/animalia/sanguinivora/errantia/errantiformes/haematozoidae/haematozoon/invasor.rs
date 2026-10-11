use crate::scene::life::biology::{Biology, Reproduction};

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.001,
    reproduction: Reproduction::Division,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 17.0;

pub const TAXON: Taxon = Taxon {
    name: "Haematozoon invasor",
    english_name: "Blood Dwellers",
    description: "Blood Dwellers are parasites that invade and develop within a host's bloodstream. \
        Developing parasites produce external juvenile growths that eventually detach to seek new hosts. \
        Their free-living stages have compact, soft bodies adapted to passive dispersal. \
        Free stages disperse passively, while most growth takes place within or immediately against a living host.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.0, 1.0, 3.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.55, 0.55, 2.3)), 0);
    for i in 0..7 {
        let z = i as f32 * 0.55 - 1.6;
        hit.join(ellipsoid(p - vec3(0.4, 0.25, z), vec3(0.3, 0.35, 0.35)), 2);
    }
    hit.join(ellipsoid(p - vec3(0.0, 0.0, -2.1), vec3(0.45, 0.4, 0.5)), 8);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.54, 0.13, 0.22, 0.0),
        vec3(0.87, 0.43, 0.46),
        [],
    )
}
