use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Social},
    capsule, louvres, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 900.0,
    lifespan_days: 18000.0,
    speed: 75.0,
    social: Social::Solitary,
    adult_mass: 14.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Aculeorhynchus acuminatus",
    "Needlesharks",
    [
        "Needlesharks carry a long, hollow lance ahead of a slim dart of a body with no fin to catch the air, and two pairs of sharply swept wings.",
        "They approach from higher air, using the updraft to recover altitude between steep, carefully aligned strikes, broad tail flukes holding each dive true.",
        "The lance concentrates force along the body axis, and a flared guard ring partway along stops it at a set depth so the animal can always draw back out.",
        "Hanging from its catch, it feeds through the lance itself while three tentacle pairs on a collar at its base wrap the prey, a ring of collar eyes sights straight down the lance, and when it has fed it pushes free.",
        "Open feeding routes suit their attack better than tangled Jungles, where a missed strike could drive the spike into living wood.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 19.0).extend(0.0);
    rig_tentacles(c, COLLAR + vec3(0.0, -0.45, 0.2), 1.7, BIOLOGY.oral_pairs);
}

/// The collar at the lance's base, where the eyes and tentacles ring it.
const COLLAR: Vec3 = vec3(0.0, 0.0, -4.6);

/// A diving lance, falling on its prey from above.
///
/// It drives a long, hollow rostrum through its prey; a flared guard
/// partway along stops the lance at a set depth, so it can always draw
/// back out. Then, hanging from its catch, it feeds through the lance itself while the three pairs of
/// tentacles on the collar at its base wrap the prey and hold it on, and at
/// last pushes free. A ring of eyes on the collar sights straight down the
/// lance at whatever it is aimed at. Behind, the body is all dive: deepest
/// at the shoulders where the great wing muscles sit, drawn into a slim,
/// supple stalk, with no fin to catch the air, and steadied by a crescent
/// of tail flukes swept back like the wings.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let grown = 0.15 + 0.85 * c.life.y * c.life.y;
    // Narrower than deep, for the dive.
    let narrow = vec3(p.x * 1.15, p.y, p.z);
    let spine = [
        vec4(0.0, 0.0, COLLAR.z + 0.3, 0.8),
        vec4(0.0, 0.18, -1.6, 1.3),
        vec4(0.0, 0.08, 2.0, 1.0),
        vec4(0.0, -0.05, 5.4, 0.42),
        vec4(0.0, -0.05, 6.9, 0.26),
    ];
    let mut body = 1e6f32;
    for i in 0..4 {
        let from = spine[i];
        let to = spine[i + 1];
        body = smooth_min(
            body,
            tapered_tail(narrow, from.truncate(), to.truncate(), from.w, to.w) / 1.15,
            0.6,
        );
    }
    let body = louvres(body, p, vec3(1.1, 1.0, 4.0));
    // The collar: a fleshy flange where the lance roots in the head.
    let collar = vec2(vec2(p.x, p.y).length() - 0.72, (p.z - COLLAR.z) * 1.4).length() - 0.32;
    let mut hit = Surface::new(smooth_min(body, collar, 0.5), 0);
    // Every animal's lance its own length, fluted along its length so it
    // slides in and drains.
    let length = 11.0 * grown * (0.85 + c.identity.y * 0.3);
    let tip = COLLAR - vec3(0.0, 0.0, length);
    let along = ((COLLAR.z - p.z) / length).clamp(0.0, 1.0);
    let flutes = (p.y.atan2(p.x) * 6.0).cos() * 0.035 * (1.0 - along);
    let lance = tapered_tail(p, COLLAR, tip, 0.52, 0.015) + flutes;
    let bore = capsule(
        p,
        COLLAR - vec3(0.0, 0.0, length * 0.3),
        tip - vec3(0.0, 0.0, 0.4),
        0.1,
    );
    hit.join(smooth_min(lance.max(-bore), collar, 0.3), 7);
    // The guard: a thin cone flaring back towards the body, like a barb's.
    let guard_at = COLLAR.z - length * 0.62;
    let radial = vec2(p.x, p.y).length();
    let lance_radius = 0.52 + (0.015 - 0.52) * 0.62;
    let flare = vec2(radial - lance_radius, p.z - guard_at);
    let edge = vec2(0.42, 0.32);
    let outward = (flare.dot(edge) / edge.length_squared()).clamp(0.0, 1.0);
    hit.join(
        (flare - edge * outward).length() - 0.05 * (1.2 - outward),
        7,
    );
    // The ring of eyes on the collar, all aimed along the lance.
    let ring = p - COLLAR;
    let around = ring.y.atan2(ring.x);
    let sector = TAU / 8.0;
    let turn = (around / sector).round() * sector;
    let eye = vec3(turn.cos() * 0.98, turn.sin() * 0.98, -0.2);
    hit.join((ring - eye).length() - 0.17, 3);
    // Crescent flukes, swept back, thin towards their tips.
    let side = p.x.abs();
    let reach = 2.8;
    let out = (side / reach).min(1.0);
    let centre = 6.6 + 1.1 * out * out;
    let chord = 0.9 * (1.0 - out * out).max(0.0).sqrt() + 0.04;
    let fluke = ((p.z - centre).abs() - chord * 0.5)
        .max(side - reach)
        .max(p.y.abs() - (0.13 * (1.0 - out) + 0.02));
    hit.distance = smooth_min(hit.distance, fluke * 0.8, 0.25);
    hit.join(
        oral_tentacles(
            p,
            c,
            COLLAR + vec3(0.0, -0.45, 0.2),
            1.7,
            BIOLOGY.oral_pairs,
        ),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.16, 0.34, 0.43, 0.0),
    vec3(0.58, 0.68, 0.74),
    [vec4(13.0, 2.8, -1.2, 0.75), vec4(5.5, 1.7, 3.2, 0.75)],
)
.hinged([vec2(1.1, 0.3), vec2(0.75, 0.15)])
.styled([vec4(1.0, 0.12, 2.0, 0.0), vec4(0.8, 0.3, 0.0, 0.0)])
.skinned(vec4(5.0, 0.4, 2.0, -0.10))
.flying(vec4(1.10, 0.30, 0.55, 1.00))
.marked(vec4(4.00, 4.00, 0.80, 0.45), vec3(0.08, 0.12, 0.18));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
