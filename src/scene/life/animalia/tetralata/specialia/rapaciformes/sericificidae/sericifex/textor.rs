pub use super::BIOLOGY;
use crate::scene::life::animalia::tetralata::oral_tentacles;

use spirv_std::glam::{Vec3, vec3, vec4};

pub use super::render_wings;
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 36.7;

pub const TAXON: Taxon = Taxon {
    name: "Sericifex textor",
    english_name: "Weavers",
    description: "Weavers are aerial engineers that spin silk between separate plants, turning loose vegetation into connected habitats. \
        Sticky strands catch small prey, while stronger structural silk helps hold fruit fields together. \
        Some build nests and collect shiny objects among their webs. \
        Their muscular oral palps handle silk and anchors, and their work creates routes and resources used by many other organisms and Næ crafters.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 6.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.3, 1.0, 2.4)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 0.0, -2.2), vec3(1.0, 0.8, 1.0)), 0);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -2.80), 0.65, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(ellipsoid(p - vec3(0.0, -0.6, 2.0), vec3(0.65, 0.6, 0.8)), 7);
    hit.join(eyes(p, 2.4, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.35, 0.24, 0.44, 0.0),
        vec3(0.83, 0.58, 0.30),
        [vec4(6.0, 2.0, 0.0, 0.27), vec4(4.0, 1.5, 3.0, 0.27)],
    )
}
