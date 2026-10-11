pub use super::BIOLOGY;

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 19.8;

pub const TAXON: Taxon = Taxon {
    name: "Scintilla scintillans",
    english_name: "Sparkies",
    description: "Sparkies are tiny, intensely luminous four-winged animals that gather into vast shimmering swarms. \
        Algae and seeds fuel their brilliant glow, making individual bodies appear as moving points of coloured light. \
        They are a nuisance to larger animals and Næ settlements but a vital food source for small aerial predators. \
        Their collective motion produces conspicuous drifting clouds of light.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(1.0, 1.0, 2.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.45, 0.5, 1.2)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 0.1, 0.7), vec3(0.38, 0.4, 0.55)), 9);
    hit.join(
        ellipsoid(p - vec3(0.0, 0.25, -1.0), vec3(0.5, 0.45, 0.45)),
        3,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.22, 0.71, 0.79, 0.65),
        vec3(0.76, 0.98, 0.42),
        [vec4(3.0, 1.3, 0.0, 0.24), vec4(2.3, 1.0, 1.2, 0.27)],
    )
}
