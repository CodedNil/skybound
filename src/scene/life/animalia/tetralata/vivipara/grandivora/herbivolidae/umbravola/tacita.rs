use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::{Biology, Litter},
    capsule, eyespots,
    genome::Gene,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::tetralata::CISTERN,
    animalia::tetralata::FLIGHT_MUSCLE,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.41,
    maturity_days: 2200.0,
    lifespan_days: 50000.0,
    adult_mass: 20.0,
    // One young to the pouch, rarely two.
    litter: Litter::new(1, 1, [0.8, 0.18, 0.02, 0.0, 0.0, 0.0]),
    gestation_days: 200.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 49.8;

/// The kite's shape, as its genes draw it.
///
/// How broad and how long in front, how long the whiskers on its leading edges, and how far its
/// streamers trail. A broad kite circles slower and longer on the same air, long whiskers feel a
/// carcass's warmth from further, and long streamers hold it steadier as it sinks onto its meal.
pub const WIDTH: Gene = Gene(38);
pub const FRONT: Gene = Gene(39);
pub const WHISKERS: Gene = Gene(40);
pub const STREAMERS: Gene = Gene(41);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Umbravola tacita",
    "Shadowings",
    [
        "Shadowings have no head and no body apart from the wing: each is one broad, velvet-dark diamond that circles the broad currents unseen.",
        "Whiskers along the leading edges feel for the disturbed, warm air around a falling carcass, and clusters of eyes at both tips and beneath watch the currents below.",
        "Mouth tentacles pull free tissue while the wing pairs match the remains' descent.",
        "Sharp, echoing cries call companions to food and warn other animals when a large predator remains nearby.",
        "Remembered routes and the always-visible polar beams guide their searches between meals that never remain in one place for long.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(11.0, 4.0, 17.0).extend(0.0);
    rig_tentacles(c, vec3(0.0, -0.8, -1.5), 1.3, ORAL_PAIRS);
}

/// A living kite, with no head and no body apart from the wing.
///
/// The whole animal is one broad diamond, thick and dark at the centre where its organs lie and
/// thinning to every edge, its velvet surface swallowing light so it circles unseen. Fine whiskers
/// along the leading edges feel for the warm, disturbed air above a carcass; clusters of eyes at
/// both tips and a ring beneath watch the currents far below. When it finds its meal it sinks onto
/// it, and the cluster of gripping tentacles slung beneath its centre tears free what it can carry,
/// while twin streamers behind hold it steady. No two kites share a shape, and a kite's young take
/// after it.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let width = 9.0 * (0.85 + c.gene(WIDTH) * 0.3);
    let front = 6.0 * (0.85 + c.gene(FRONT) * 0.3);
    let back = 9.0;
    let along = if p.z < 0.0 { -p.z / front } else { p.z / back };
    let shape = p.x.abs() / width + along;
    let thickness = 1.7 * (1.0 - shape).max(0.0).powf(1.5) + 0.08;
    let dihedral = p.x.abs() / width * 1.3 + (p.x.abs() * 0.4 - c.animation.z).sin() * 0.08;
    let plate = ((p.y - dihedral).abs() - thickness * 0.5).max((shape - 1.0) * width * 0.4) * 0.6;
    let mut hit = Surface::new(plate, 0);
    let side = p.x.abs();
    let spacing = width / 9.0;
    let slot = (side / spacing).round().clamp(1.0, 8.0) * spacing;
    let edge = vec3(slot, slot / width * 1.3, -(1.0 - slot / width) * front);
    let whisker = capsule(
        vec3(side, p.y, p.z),
        edge,
        edge + vec3(0.25, 0.1, -0.9) * (0.6 + c.gene(WHISKERS) * 0.8),
        0.03,
    );
    hit.blend(whisker, 1, 0.03);
    hit.socket(
        eyespots(p, vec3(width * 0.93, 1.25, -0.3), 0.3, 3, 0.12),
        0.12,
    );
    hit.socket(eyespots(p, vec3(0.9, -0.95, -0.8), 0.4, 3, 0.16), 0.16);
    let streamer = tapered_tail(
        vec3(side, p.y, p.z),
        vec3(0.7, 0.0, back * 0.9),
        vec3(
            1.6 + (c.animation.z * 0.8).sin() * 0.4,
            -0.6,
            back + 4.5 + c.gene(STREAMERS) * 2.5,
        ),
        0.2,
        0.02,
    );
    hit.blend(streamer, 1, 0.15);
    hit.blend(
        oral_tentacles(p, c, vec3(0.0, -0.8, -1.5), 1.3, ORAL_PAIRS),
        8,
        0.1,
    );
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.11, -3.49), vec3(0.08, 0.08, 0.08)),
    site(PUMP, vec3(0.00, -0.04, -1.37), vec3(0.28, 0.13, 0.28)),
    site(CROP, vec3(0.00, -0.04, 1.06), vec3(0.34, 0.26, 0.34)),
    pair(BREATHER, vec3(2.95, 0.50, -1.52), vec3(0.13, 0.13, 0.13)),
    site(FLOAT, vec3(0.00, 0.46, 0.76), vec3(0.15, 0.15, 0.15)),
    pair(
        FLIGHT_MUSCLE,
        vec3(2.55, 0.36, -0.30),
        vec3(0.20, 0.13, 0.20),
    ),
    site(CISTERN, vec3(0.00, -0.04, 2.88), vec3(0.23, 0.13, 0.23)),
    site(BROOD, vec3(0.00, 0.04, 4.10), vec3(0.17, 0.13, 0.17)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.06, 0.08, 0.14, 0.0),
    vec3(0.16, 0.23, 0.29),
    [Vec4::ZERO; 2],
)
.hinged([Vec2::ZERO, Vec2::ZERO])
.styled([vec4(0.3, 0.9, 0.0, 0.7), vec4(0.3, 0.9, 0.0, 0.7)])
.skinned(vec4(3.0, 1.0, 1.4, 0.20))
.brooding(vec4(0.6, 2.2, 1.4, 0.9));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology, Take};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        eats: |_, _| false,
        takes: Take::Carrion,
        gleans: true,
        speed: 32.0,
        ..super::super::PHYSIOLOGY
    };
}
