use crate::scene::life::Tetralate;

pub const BIOLOGY: Biology = Biology {
    size: 0.41,
    maturity_days: 2200.0,
    lifespan_days: 50000.0,
    adult_mass: 20.0,
    // One young to the pouch, rarely two.
    litter: Litter::new(1, 1, [0.8, 0.18, 0.02, 0.0, 0.0, 0.0]),
    gestation_days: 200.0,
    // Circling high over the routes others travel, watching the currents below.
    altitude: (3000.0, 13000.0),
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
pub use nature::{ETHOLOGY, PHYSIOLOGY};

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

pub fn animate(c: &mut Tetralate) {
    c.opaque_bounds = vec3(11.0, 4.0, 17.0).extend(0.0);
    rig_tentacles(c, vec3(0.0, -0.8, -1.5), 1.3, ORAL_PAIRS);
}

/// Its rows of eyes, each as [`eyespots`] lays them out: one along the upper edge of a body
/// `width` across, and one beneath.
const fn upper(width: f32) -> (Vec3, f32, u32) {
    (vec3(width * 0.93, 1.25, -0.3), 0.3, 3)
}
const LOWER: (Vec3, f32, u32) = (vec3(0.9, -0.95, -0.8), 0.4, 3);

/// How broad its body is, as its genes have it.
fn width(c: &CreatureInstance) -> f32 {
    9.0 * (0.85 + c.gene(WIDTH) * 0.3)
}

/// Its eyes, the upper row and then the lower, each seated on its own skin.
#[cfg(not(target_arch = "spirv"))]
pub const ANCHORS: u32 = 6;

/// Where eye `index` lies on its design, before it is seated on its own skin.
#[cfg(not(target_arch = "spirv"))]
pub fn anchor(index: u32, c: &CreatureInstance) -> Vec4 {
    if index < 3 {
        let (first, spacing, _) = upper(width(c));
        spot(first, spacing, index).extend(spot_radius(0.12, index) * SUNK)
    } else {
        spot(LOWER.0, LOWER.1, index - 3).extend(spot_radius(0.16, index - 3) * SUNK)
    }
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
pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
    let width = width(c);
    let front = 6.0 * (0.85 + c.gene(FRONT) * 0.3);
    let back = 9.0;
    let along = if p.z < 0.0 { -p.z / front } else { p.z / back };
    let shape = p.x.abs() / width + along;
    let thickness = 1.7 * (1.0 - shape).max(0.0).powf(1.5) + 0.08;
    let dihedral = p.x.abs() / width * 1.3 + (p.x.abs() * 0.4 - c.animation.z).sin() * 0.08;
    let plate = ((p.y - dihedral).abs() - thickness * 0.5).max((shape - 1.0) * width * 0.4) * 0.6;
    // The breathing louvres every tetralate has, in the thick middle of the wing where the body
    // once was.
    let plate = louvres(plate, p, vec3(3.0, 2.0, 6.0));
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
    // Measured bare, the body has no eyes set in it yet.
    if !c.wings.measuring() {
        hit.socket(seated_spots(p, c, 0, upper(width), 0.12), 0.12);
        hit.socket(seated_spots(p, c, 3, LOWER, 0.16), 0.16);
    }
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

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.06, 0.08, 0.14, 0.0),
    vec3(0.16, 0.23, 0.29),
    [Vec4::ZERO; 2],
)
.hinged([Vec2::ZERO, Vec2::ZERO])
.stalked([vec2(0.50, 0.25), vec2(0.45, 0.25)])
.styled([vec4(0.3, 0.9, 0.0, 0.7), vec4(0.3, 0.9, 0.0, 0.7)])
.skinned(vec4(3.0, 1.0, 1.4, 0.20))
.brooding(vec4(0.6, 2.2, 1.4, 0.9));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::{
        mind::Ethology,
        nature::SCAVENGE,
        physiology::{Light, Physiology, Take},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        eats: |_, _| false,
        takes: Take::Carrion,
        gleans: true,
        speed: 32.0,
        ..super::super::PHYSIOLOGY
    };
    /// A Shadowing that finds a carcass cries out as it circles down, and its companions come from
    /// far off to share it.
    pub const ETHOLOGY: Ethology = Ethology {
        act: |mind, s| {
            let goal = (super::super::ETHOLOGY.act)(mind, s);
            if mind.intent == SCAVENGE {
                return Some(goal.unwrap_or_else(|| (SCAVENGE.act)(mind, s)).calling());
            }
            goal
        },
        ..super::super::ETHOLOGY
    };
}
