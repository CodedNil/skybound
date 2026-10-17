use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 12.0,
    lifespan_days: 120.0,
    ..super::BIOLOGY
};
use crate::scene::life::animalia::tetralata::oral_tentacles;

use spirv_std::glam::{Vec3, vec3, vec4};

pub use super::render_wings;
use crate::scene::life::{CreatureInstance, ellipsoid};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 21.6;

taxon!(
    Species,
    "Manduca gregaria",
    "Nibblers",
    [
        "Nibblers are tiny four-winged scavengers whose compact bodies turn quickly around suspended organic debris.",
        "Feeding swarms gather where damaged plants, carcasses or crowded grazing routes release loose material.",
        "Each animal follows nearby motion and food cues, producing coordinated murmurations without a fixed leader.",
        "Rapid growth lets a swarm exploit a short-lived concentration before the current scatters it.",
        "By intercepting debris before it falls toward the aur ocean, they keep nutrients circulating among the Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 4.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.8, 0.65, 1.5)), 0);
    hit.join(
        ellipsoid(p - vec3(0.0, 0.2, -1.25), vec3(0.7, 0.6, 0.55)),
        3,
    );
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -1.90), 0.40, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.47, 0.26, 0.18, 0.0),
        vec3(0.86, 0.56, 0.28),
        [vec4(3.5, 1.2, 0.0, 0.3), vec4(2.2, 1.0, 1.7, 0.35)],
    )
}
