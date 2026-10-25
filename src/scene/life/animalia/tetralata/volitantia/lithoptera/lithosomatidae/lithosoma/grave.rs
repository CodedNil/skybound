use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    annulated,
    biology::Biology,
    rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1800.0,
    lifespan_days: 60000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 46.8;

taxon!(
    Species,
    "Lithosoma grave",
    "Grounders",
    [
        "Grounders carry heavy mineral armour over dense bodies supported by broad lifting surfaces.",
        "They move slowly through the lower Gardens, grazing Root-Stars while staying above the lethal Veiland mists.",
        "Wide wings brake descent through strong updrafts, so the animals favour reliable rising columns rather than weak or rapidly shifting air.",
        "Mineral-rich currents renew the crust but expose them to dusty haze and the danger of nearby geyser jets.",
        "Their feeding routes skirt those jets, and the dim aur phase forces a careful balance between remaining near food and maintaining safe altitude.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(12.0, 10.0, 14.0).extend(0.0);
    rig_tentacles(c, vec3(0.0, -1.3, -12.3), 4.0, BIOLOGY.oral_pairs);
}

/// A heavy, stone-backed flyer: a broad, ringed body roofed by
/// overlapping plates like a woodlouse, a blunt armoured head with deep-set
/// eyes, and thick mouth tentacles beneath.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut body = annulated(
        p,
        vec3(0.0, 0.0, -7.0),
        vec3(0.0, 0.6, 9.5),
        vec3(4.6, 7.6, 1.6),
        8.0,
        0.12,
        0.66,
    );
    for i in 0..7 {
        let f = i as f32;
        let z = f * 2.35 - 6.6;
        let span = 7.0 - (f - 2.5).abs() * 0.55;
        let plate = rounded(
            p - vec3(0.0, 3.3 - (z * 0.12).powi(2), z),
            vec3(span, 2.0, 1.55),
        );
        body = smooth_min(body, plate, 0.35);
    }
    let head = rounded(p - vec3(0.0, -0.4, -9.6), vec3(4.0, 3.2, 3.2));
    let mut hit = Surface::new(smooth_min(body, head, 1.2), 5);
    let mirrored = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        rounded(mirrored - vec3(2.55, 0.75, -11.75), vec3(0.5, 0.42, 0.55)),
        3,
    );
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -1.3, -12.3), 4.0, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.24, 0.24, 0.29, 0.0),
    vec3(0.46, 0.32, 0.56),
    [vec4(20.0, 5.5, -2.5, 0.98), vec4(14.0, 4.0, 3.5, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
