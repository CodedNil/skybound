use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 18000.0,
    lifespan_days: 365_000.0,
    speed: 8.0,
    adult_mass: 200.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::{CreatureInstance, eyes, prepare_pose, ribbons, tapered_tail, torso};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 55.6;

taxon!(
    Species,
    "Vocigenus antiquus",
    "Old Wise Ones",
    [
        "Old Wise Ones are rare giants with expansive wings and long white fibres hanging beneath the head like a flowing beard.",
        "Slow wing strokes and careful use of rising air carry their immense bodies between kelp forests and Laevensails.",
        "Three pairs of mouth tentacles gather only those soft plants, giving the lineage a narrow dependence on healthy floating forests.",
        "Rich calls pass between companions long before their pale bodies become visible through the haze.",
        "Their young develop slowly within a live-bearing social lineage whose journeys preserve access to feeding areas across many aur cycles.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 10.0, 17.0, 5);
    c.opaque_bounds = vec3(10.0, 12.0, 27.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(6.0, 6.0, 12.0), 1.8), 0);
    hit.join(eyes(p, 12.0, 1.8), 3);
    for i in -3..=3 {
        let x = i as f32 * 0.7;
        hit.join(
            tapered_tail(
                p,
                vec3(x, -1.0, -13.0),
                vec3(x * 1.5, -10.0 + x.abs(), -7.0),
                0.65,
                0.12,
            ),
            7,
        );
    }
    hit.join(ribbons(p, c, 5), 7);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.5, -17.0), 2.6, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.75, 0.73, 0.64, 0.0),
        vec3(0.94, 0.92, 0.83),
        [vec4(26.0, 8.0, 0.0, 0.98), vec4(18.0, 6.0, 12.0, 0.98)],
    )
}
