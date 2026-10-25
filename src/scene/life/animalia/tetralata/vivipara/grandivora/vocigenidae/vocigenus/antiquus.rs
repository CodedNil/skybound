use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth, prepare_pose, ribbons, rig_ribbons, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 18000.0,
    lifespan_days: 365_000.0,
    speed: 8.0,
    adult_mass: 200.0,
    ..super::BIOLOGY
};

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
    rig_tentacles(c, mouth(12.5, 2.1), 2.6, BIOLOGY.oral_pairs);
    // Display streamers off the belly, flown for companions across open air.
    rig_ribbons(c, 5, vec3(0.0, -4.6, 3.0), 1.4, 18.0);
}

/// An ancient, enormous singer: a massive, wrinkled body with a throat sac
/// slung beneath the jaw, a beard of long, stiff barbels, and trailing
/// ribbons.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let folds = (p.z * 1.3 + (p.x * 0.6).sin()).sin() * 0.12;
    let body = torso(p, vec3(5.4, 5.4, 12.5), 2.1) + folds;
    let sac = rounded(p - vec3(0.0, -4.2, -11.5), vec3(3.6, 3.3, 4.2));
    let mut hit = Surface::new(smooth_min(body, sac, 1.6), 0);
    hit.join(eyes(p, 12.5, 2.1), 3);
    for i in -3..=3 {
        let x = i as f32 * 0.75;
        hit.join(
            tapered_tail(
                p,
                vec3(x, -1.6, -13.2),
                vec3(x * 1.5, -9.5 + x.abs(), -8.5),
                0.55,
                0.08,
            ),
            7,
        );
    }
    hit.join(ribbons(p, c, 5), 7);
    hit.join(
        oral_tentacles(p, c, mouth(12.5, 2.1), 2.6, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.75, 0.73, 0.64, 0.0),
    vec3(0.94, 0.92, 0.83),
    [vec4(26.0, 8.0, -3.5, 0.98), vec4(18.0, 6.0, 4.5, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
