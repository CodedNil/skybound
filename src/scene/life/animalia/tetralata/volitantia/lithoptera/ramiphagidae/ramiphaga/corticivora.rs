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
use crate::scene::{
    life::{
        CreatureInstance, Look,
        animalia::tetralata::{ORAL_PAIRS, curved_tube, oral_tentacles, rig_tentacles},
        biology::Biology,
        genome::Gene,
        surface::{Ray, Surface, TraceResult},
    },
    math::Power,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.33,
    maturity_days: 400.0,
    lifespan_days: 12000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 26.8;

/// How deeply banded its armour, how heavy its jaws, and how thick its curl.
///
/// Deep bands flex further round a thick branch, heavy jaws crush harder bark, and a thick curl
/// grips harder but weighs more in flight.
pub const PLATING: Gene = Gene(38);
pub const JAWS: Gene = Gene(39);
pub const GIRTH: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Ramiphaga corticivora",
    "Driftmunchers",
    [
        "Driftmunchers curl their armoured bodies into a C around a drifting Bramble branch, locked shut by a hooked tail claw, and crush its bark and thorns with heavy jaws facing inward from the front of the curl.",
        "Reinforced mandibles work beside the inherited mouth appendages, which hold a branch without functioning as walking limbs.",
        "They strip outer growth in small patches so that feeding does not immediately destroy the framework supporting them.",
        "Protective tissues tolerate brief contact with charged spines, but the soft underside remains vulnerable during a careless turn.",
        "Their feeding opens resistant woody nutrients to smaller scavengers and leaves sheltered gaps used by other inhabitants of the tangle.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.5, 4.0).extend(0.0);
    rig_tentacles(c, head(0.0) + vec3(0.0, 0.35, 0.55), 0.9, ORAL_PAIRS);
}

/// A bark-eater armoured in banded plates, with a blunt head, small deep eyes and a pair of heavy
/// gouging tusks. The radius of the curl, and how far round it runs either side of the top, leaving
/// a gap below for the branch.
const CURL: f32 = 2.2;

const REACH: f32 = 2.4;

/// A point on the curl at an angle from its top, `inward` towards the branch it holds.
fn on_curl(angle: f32, inward: f32) -> Vec3 {
    let radius = CURL - inward;
    vec3(0.0, radius * angle.cos(), -radius * angle.sin())
}

/// The head, at the curl's front end; `inward` moves towards the branch.
fn head(inward: f32) -> Vec3 {
    on_curl(REACH, inward)
}

/// A living clamp.
///
/// Its armoured body curls into a C round the branch it is eating, banded like a ring of plates, so
/// no legs are needed to hold on: the curl itself is the grip, locked shut by a hooked claw at the
/// tail. At the front end heavy crushing jaws face inward, onto the branch inside the curl,
/// stripping its bark a patch at a time while the short mouth tentacles hold fast beside them. The
/// soft underside lines the inside of the C, pressed against the wood, where only a careless turn
/// exposes it to the Bramble's charged thorns. Its wings are compact, stiff plates set on the top
/// of the curl.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let angle = (-p.z).atan2(p.y).clamp(-REACH, REACH);
    let spine = on_curl(angle, 0.0);
    let middle = 1.0 - (angle / REACH).power(2);
    let band = (angle * 7.0).cos();
    let radius = (0.55 + 0.45 * middle)
        * (0.9 + c.gene(GIRTH) * 0.2)
        * (1.0 - (0.03 + c.gene(PLATING) * 0.1) * band * band);
    let body = ((p - spine).length() - radius) * 0.8;
    // Plates outside, soft tissue on the inside of the curl.
    let inward = vec2(p.y, p.z).length() < CURL - radius * 0.35;
    let mut hit = Surface::new(body, if inward { 8 } else { 5 });
    let side = vec3(p.x.abs(), p.y, p.z);
    let jaw = head(0.0) + vec3(0.28, 0.0, 0.0);
    hit.blend(
        curved_tube(
            side,
            jaw,
            jaw + (head(1.0) - head(0.0)) * 0.9 + vec3(0.2, 0.0, 0.0),
            head(1.6) + vec3(0.05, 0.25, 0.0),
            0.26 + c.gene(JAWS) * 0.14,
        ),
        7,
        0.1,
    );
    let tail = on_curl(-REACH, 0.0);
    hit.blend(
        curved_tube(
            p,
            tail,
            tail + (on_curl(-REACH, 0.9) - tail) * 0.8 + vec3(0.0, -0.4, 0.0),
            on_curl(-REACH + 0.25, 1.3),
            0.2,
        ),
        7,
        0.1,
    );
    // Small, deep eyes set into the head's outer plates.
    let eye = head(-0.45) + vec3(0.4, -0.1, -0.15);
    hit.socket((side - eye).length() - 0.13, 0.13);
    hit.blend(
        oral_tentacles(p, c, head(0.0) + vec3(0.0, 0.35, 0.55), 0.9, ORAL_PAIRS),
        8,
        0.1,
    );
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 1.17, -2.04), vec3(0.15, 0.41, 0.48)),
    site(PUMP, vec3(0.00, 0.60, -1.32), vec3(0.10, 0.10, 0.10)),
    site(CROP, vec3(0.00, 1.17, 0.96), vec3(0.18, 0.18, 0.18)),
    pair(BREATHER, vec3(0.37, 1.04, -1.21), vec3(0.11, 0.11, 0.11)),
    site(FLOAT, vec3(0.00, 2.21, 0.11), vec3(0.19, 0.54, 0.59)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.37, 1.69, -0.50),
        vec3(0.15, 0.26, 0.26),
    ),
    site(CISTERN, vec3(0.00, 0.44, 1.46), vec3(0.10, 0.10, 0.10)),
    site(BROOD, vec3(0.00, 0.84, 1.96), vec3(0.17, 0.44, 0.53)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.34, 0.22, 0.13, 0.0),
    vec3(0.56, 0.36, 0.17),
    [vec4(3.4, 1.6, -0.2, 0.9), vec4(2.2, 1.1, 0.8, 0.9)],
)
.hinged([vec2(0.4, 3.0), vec2(0.4, 2.9)])
.styled([vec4(0.2, 1.0, 0.0, 0.0), vec4(0.2, 1.0, 0.0, 0.0)])
.skinned(vec4(2.0, 1.6, 1.4, 0.15))
.flying(vec4(2.00, 0.30, 0.10, 0.60))
.marked(vec4(1.00, 3.00, 0.50, 0.45), vec3(0.25, 0.16, 0.08));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
