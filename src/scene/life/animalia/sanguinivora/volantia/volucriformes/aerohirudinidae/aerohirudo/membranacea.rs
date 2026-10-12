use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 60.0,
    lifespan_days: 1800.0,
    speed: 18.0,
    adult_mass: 0.06,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec2, Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 24.2;

pub const TAXON: Taxon = Taxon {
    name: "Aerohirudo membranacea",
    english_name: "Air Leeches",
    description: "Air Leeches carry large, translucent, bat-like membrane wings around a narrow flexible body. \
        Fine supporting structures remain visible through the wing surfaces, while the feeding end bears a rapid sucker grip. \
        They hover near busy feeding grounds and ambush passing animals or plants. \
        A brief attachment draws blood or sap before they release and withdraw into cover.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.4, 1.4, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.75, 0.6, 4.6)), 0);
    let ring = Vec3::new(p.x, p.y - 0.15, p.z + 4.5);
    hit.join(
        (Vec2::new(ring.x, ring.y).length() - 0.6).hypot(ring.z) - 0.14,
        8,
    );
    hit.join(eyes(p, 4.5, 0.3), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.29, 0.13, 0.19, 0.0),
        vec3(0.75, 0.27, 0.33),
        [vec4(8.0, 3.2, 0.0, 0.25)],
    )
}
