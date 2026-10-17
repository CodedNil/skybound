use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 12.0,
    lifespan_days: 240.0,
    feeding: Feeding::Grazer,
    social: Social::Mutualist,
    adult_mass: 0.002,
    speed: 1.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 16.0;

taxon!(
    Species,
    "Candela lucens",
    "Glowgnomes",
    [
        "Glowgnomes are tiny luminous sap feeders that inhabit sheltered burrows in Whisperwood.",
        "They retain the single wing pair of their parasitic ancestors, but coordinated wing strokes now help draw the host toward favourable currents.",
        "Pheromone signals carried through six vertical sap veins recruit groups on different sides of the trunk.",
        "Their light draws visitors through nearby flowering growth during the dim aur phase, while sap and wood cavities provide food and shelter.",
        "Limited feeding and shared movement have turned an ancestral parasitic visit into a partnership on which the drifting tree depends.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 3.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 1.6, 1.0)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 1.3, 0.0), vec3(0.85, 0.9, 0.85)), 9);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -1.40), 0.35, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.55, 0.32, 0.15, 0.0),
        vec3(1.0, 0.72, 0.20),
        [vec4(3.0, 1.3, 0.0, 0.32)],
    )
}
