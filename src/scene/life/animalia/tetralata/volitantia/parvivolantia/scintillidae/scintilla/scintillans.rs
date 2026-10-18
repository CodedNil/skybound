use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::biology::Biology;
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 3.0,
    lifespan_days: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 19.8;

taxon!(
    Species,
    "Scintilla scintillans",
    "Sparkies",
    [
        "Sparkies appear as moving points of intense colour as vast swarms gather algae and suspended seeds.",
        "Two pairs of quick wings let each animal adjust to its neighbours and to small changes in the feeding current.",
        "Their light preserves contact through dim aur phases, but also draws Shimmerays and other small-prey hunters.",
        "Fast growth allows dense swarms to appear around a brief food pulse and disperse before the current empties.",
        "Such gatherings trouble larger animals and Næ settlements while supplying an essential link between microscopic plant life and aerial predators.",
    ],
);

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
