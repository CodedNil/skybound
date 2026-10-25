use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyespots, prepare_pose, ribbons, rig_ribbons, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
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
        "Old Wise Ones are rare giants whose long, pale bodies carry a chain of swelling gas bladders along the back, with long white barbels hanging beneath the blunt front like a flowing beard.",
        "Slow wing strokes and careful use of rising air carry their immense bodies between kelp forests and Laevensails.",
        "Three pairs of mouth tentacles gather only those soft plants, giving the lineage a narrow dependence on healthy floating forests.",
        "Rich calls, shaped by a throat sac and a double rank of hollow resonating horns along the back, pass between companions long before their pale bodies become visible through the haze.",
        "Their young develop slowly within a live-bearing social lineage whose journeys preserve access to feeding areas across many aur cycles.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 12.0, 17.0, 5);
    c.opaque_bounds = vec3(10.0, 12.0, 30.0).extend(0.0);
    rig_tentacles(c, vec3(0.0, -2.4, -15.2), 2.6, BIOLOGY.oral_pairs);
    // Display streamers off the belly, flown for companions across open air.
    rig_ribbons(c, 5, vec3(0.0, -4.4, 3.0), 1.4, 18.0);
}

/// A living airship.
///
/// Its long, pale body carries a chain of gas bladders swelling along the back,
/// so a giant that lives for centuries can idle on slow strokes between distant
/// floating forests; between them rise the hollow horns that shape its calls,
/// carried across open sky long before the animal itself is visible. Beneath
/// its blunt front hangs a beard of long white barbels and the throat sac it
/// sings through, the three tentacle pairs that gather its soft plants, and
/// rows of small eyes along each flank. Its bladders swell and its horns
/// lengthen through its long life, so no two elders are alike.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let folds = (p.z * 1.3 + (p.x * 0.6).sin()).sin() * 0.12;
    let mut body = rounded(p, vec3(5.0, 4.4, 15.0)) + folds;
    let sac = rounded(p - vec3(0.0, -4.0, -11.0), vec3(3.4, 3.0, 4.0));
    body = smooth_min(body, sac, 1.6);
    for i in 0..5 {
        let f = i as f32;
        let swell = 1.0 + (f * 12.9898 + c.identity.x * 311.7).sin() * 0.25;
        let z = f * 5.0 - 10.0;
        let rise = 3.8 - (z / 15.0).powi(2) * 2.0;
        let bladder = rounded(p - vec3(0.0, rise, z), vec3(3.2, 2.2, 2.5) * swell);
        body = smooth_min(body, bladder, 1.2);
    }
    let mut hit = Surface::new(body, 0);
    let back = vec3(p.x.abs(), p.y, p.z);
    for i in 0..4 {
        let z = -7.5 + i as f32 * 5.0;
        let length = (6.0 - i as f32 * 1.2) * (0.7 + c.life.z * 0.3 + c.identity.w * 0.3);
        let foot = vec3(2.8, 4.0 - (z / 15.0).powi(2) * 2.0, z);
        let tip = foot + vec3(0.8, length, length * 0.4);
        let horn = tapered_tail(back, foot, tip, 0.8, 0.6);
        let bore = tapered_tail(
            back,
            foot + (tip - foot) * 0.3,
            tip + (tip - foot) * 0.05,
            0.48,
            0.44,
        );
        hit.join(horn.max(-bore), 7);
    }
    for i in -3..=3 {
        let x = i as f32 * 0.8;
        hit.join(
            tapered_tail(
                p,
                vec3(x, -2.6, -14.0),
                vec3(x * 1.5, -11.0 + x.abs(), -10.0),
                0.5,
                0.07,
            ),
            7,
        );
    }
    hit.join(eyespots(p, vec3(4.2, 0.4, -12.0), 1.1, 4, 0.32), 3);
    hit.join(ribbons(p, c, 5), 7);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -2.4, -15.2), 2.6, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.75, 0.73, 0.64, 0.0),
    vec3(0.94, 0.92, 0.83),
    [vec4(26.0, 8.0, -3.5, 0.98), vec4(18.0, 6.0, 4.5, 0.98)],
)
.hinged([vec2(5.0, 1.5), vec2(4.6, 1.3)])
.styled([vec4(0.4, 0.9, 6.0, 0.0), vec4(0.5, 0.8, 4.0, 0.0)])
.skinned(vec4(1.0, 1.9, 0.6, 0.10))
.flying(vec4(0.18, 0.40, 0.45, 1.30))
.marked(vec4(4.00, 4.00, 0.40, 0.42), vec3(0.98, 0.97, 0.92))
.brooding(vec4(2.8, 12.0, 2.2, 1.6));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
