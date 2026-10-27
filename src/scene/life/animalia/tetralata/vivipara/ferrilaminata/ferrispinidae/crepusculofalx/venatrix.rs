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
    organs::{Function, Organ, Viscera, pair, site},
};
use crate::scene::{
    life::{
        CreatureInstance, Look,
        animalia::tetralata::{gape, oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
        biology::Biology,
        ellipsoid,
        genome::Gene,
        louvres, prepare_pose, rounded, smooth_min,
        surface::{Ray, Surface, TraceResult},
        tapered_tail,
    },
    math::Power,
};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.19,
    maturity_days: 730.0,
    lifespan_days: 14600.0,
    adult_mass: 5.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

/// How many glowing threads its lure trails, how far below it the lure dangles, how thick the fur
/// of its ruff, and how large its great eyes.
///
/// A full lure draws prey from further, a low one keeps the hunter further from what it draws, a
/// thick ruff hides its outline better, and large eyes see prey sooner against the glow.
pub const LURE_THREADS: Gene = Gene(38);
pub const DANGLE: Gene = Gene(39);
pub const RUFF: Gene = Gene(40);
pub const EYES: Gene = Gene(41);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Crepusculofalx venatrix",
    "Duskslicers",
    [
        "Duskslicers hang almost motionless in the mineral haze on broad, dust-furred wings, their mottled bodies lost in it, a glowing lure dangling far below on a whip of a tail.",
        "Small animals drawn up to the lure's apparent feeding signal come within reach of its long mouth tentacles, which shoot out and seize them, and it hangs on in the haze to feed.",
        "Patches of gathered iron armour its back and the leading edges of its forewings against the hunters that come for so still a target, more and broader the older it is.",
        "A pair of huge eyes on the underside of the head looks down at the faint glow of the aur, against which prey shows as a shadow even in the dimmest equatorial skies, where they live.",
        "Fringed hindwings hush the glide, and mouth tentacles gather the meal; scavenging carries them through haze too thick for the lure to be seen.",
        "Their dim-field specialisation makes clear, brightly lit Gardens less favourable than the mineral-rich currents that many other animals only visit.",
    ],
);

/// Where the mouth tentacles hang, below the head.
const MOUTH: Vec3 = vec3(0.0, -0.75, -4.1);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.2, 18.0, 0);
    // The tail hangs, so its lure dangles below the hunter.
    for i in 0..6 {
        let t = i as f32 / 5.0;
        c.tail[i].y -= t * t * (6.2 + c.gene(DANGLE) * 1.6);
        c.tail[i].z -= t * t * 4.0;
    }
    c.opaque_bounds = vec3(6.0, 9.0, 21.0).extend(0.0);
    rig_tentacles(c, MOUTH, 0.8, ORAL_PAIRS);
}

/// A haze-hanging ambusher, built to be unseen until the cut.
///
/// A short, broad, flattened body is furred to hold the dust it hangs in, so it takes the haze's
/// own colour; a blunt head has two great eyes on its underside, looking down for prey against the
/// aur's glow; and a long, thin whip of a tail hangs below, ending in a lure of glowing threads.
/// The weapon is in the wings: the forewings' leading edges are iron, curving to a scythe point.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let bulk = vec3(1.9, 0.95, 3.2);
    let body = rounded(p, bulk);
    let head = rounded(p - vec3(0.0, -0.1, -3.4), vec3(1.25, 0.85, 1.15));
    let mut d = louvres(smooth_min(body, head, 0.7), p, bulk);
    // A ruff of longer fur over the shoulders, breaking up its outline.
    let fur = c.gene(RUFF);
    let ruff = ellipsoid(
        p - vec3(0.0, 0.35, -1.9),
        vec3(2.1, 0.6 + fur * 0.3, 0.8 + fur * 0.2),
    ) + (p.x * 9.0).sin() * (p.z * 7.0).sin() * (0.04 + fur * 0.04);
    d = smooth_min(d, ruff, 0.4);
    // A thin whip of a tail.
    let mut whip = 1e6f32;
    for i in 0..5 {
        let t = (i + 1) as f32 / 5.0;
        whip = whip.min(tapered_tail(
            p,
            c.tail[i].truncate(),
            c.tail[i + 1].truncate(),
            0.55 * (1.1 - t) + 0.05,
            0.55 * (0.9 - t) + 0.05,
        ));
    }
    let skin = smooth_min(d, whip, 0.5);
    let mut hit = Surface::new(skin, 0);
    // Scales of gathered iron over the back, few and scattered on a young one, crowding together on
    // an old one that has gathered much.
    let iron = c.stores.x;
    if iron > 0.02 && p.y > 0.1 && p.z < 2.6 {
        let cell = (vec2(p.x, p.z) * 2.5).round();
        let spot =
            ((cell.dot(vec2(12.9898, 78.233)) + c.identity.x * 37.0).sin() * 43_758.547).fract_gl();
        if spot < iron * 0.85 {
            let within = (vec2(p.x, p.z) * 2.5 - cell).length() / 2.5 - 0.17;
            hit.blend((skin - 0.06).max(within), 6, 0.03);
        }
    }
    // The lure: a bulb trailing glowing threads.
    let end = c.tail[5].truncate();
    let mut lure = ellipsoid(p - end, vec3(0.35, 0.45, 0.35));
    let threads = 2 + (c.gene(LURE_THREADS) * 3.99) as u32;
    for thread in 0..6 {
        if thread >= threads {
            break;
        }
        let f = (thread as f32 - (threads as f32 - 1.0) * 0.5) * 2.0 / threads as f32;
        let sway = (c.animation.z * 1.7 + f * 2.0).sin() * 0.3;
        lure = lure.min(tapered_tail(
            p,
            end,
            end + vec3(f * 0.5 + sway, -1.6 + f.abs() * 0.4, 0.4),
            0.08,
            0.02,
        ));
    }
    hit.join(lure, 9);
    // Two great eyes beneath the head, set to look down and a little ahead.
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        ellipsoid(
            side - vec3(0.6, -0.55, -3.75),
            vec3(0.55, 0.5, 0.6) * (0.85 + c.gene(EYES) * 0.3),
        ),
        3,
    );
    maw(&mut hit, p, c);
    hit.join(oral_tentacles(p, c, MOUTH, 0.8, ORAL_PAIRS), 8);
    hit
}

/// The maw: a wide mouth across the front of the blunt head, closed to a thin seam in the haze, so
/// nothing about the hunter's face gives it away, and gaping wide when it feeds on what its wings
/// have cut down. Rows of fine, inward-hooked teeth line both jaws and hold the meat its tentacles
/// feed to it.
fn maw(hit: &mut Surface, p: Vec3, c: &CreatureInstance) {
    let q = p - vec3(0.0, -0.32, -4.35);
    if q.length() > 1.6 {
        return;
    }
    let open = gape(c);
    // Deepest at the front, the jaws meeting further back at the corners.
    let height = 0.03 + open * 0.38 * (1.0 - (q.x / 0.95).power(2)).max(0.0);
    let cavity = ellipsoid(q - vec3(0.0, 0.0, 0.35), vec3(0.95, height, 0.75));
    hit.carve(cavity, 0.06);
    // Teeth along both lips, hooked in towards the throat.
    let spacing = 0.16;
    let cell = (q.x / spacing).round().clamp(-5.0, 5.0) * spacing;
    let front = -0.38 + (cell / 0.95).power(2) * 0.4;
    for jaw in 0..2 {
        let side = if jaw == 0 { 1.0 } else { -1.0 };
        let root = vec3(cell, side * height, front);
        let tip = root + vec3(0.0, -side * (0.05 + open * 0.1), 0.06);
        hit.join(tapered_tail(q, root, tip, 0.035, 0.006), 7);
    }
}

/// The lure gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const LURE: Organ = Organ::new("lure gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.08, -3.56), vec3(0.33, 0.16, 0.46)),
    site(PUMP, vec3(0.00, -0.08, -2.06), vec3(0.35, 0.22, 0.70)),
    site(CROP, vec3(0.00, -0.24, -0.19), vec3(0.49, 0.34, 0.49)),
    pair(BREATHER, vec3(0.80, 0.16, -2.62), vec3(0.22, 0.19, 0.22)),
    site(FLOAT, vec3(0.00, 0.40, -0.56), vec3(0.38, 0.22, 0.38)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.80, 0.16, -1.12),
        vec3(0.22, 0.18, 0.22),
    ),
    site(CISTERN, vec3(0.00, -0.08, 1.31), vec3(0.35, 0.33, 0.80)),
    site(BROOD, vec3(0.00, -0.08, 2.44), vec3(0.35, 0.18, 0.53)),
    site(LURE, vec3(0.00, -4.90, 13.69), vec3(0.05, 0.05, 0.05)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.36, 0.32, 0.27, 0.0),
    vec3(0.72, 0.56, 0.34),
    [vec4(12.0, 3.6, -1.3, 0.85), vec4(7.5, 3.4, 1.9, 0.75)],
)
.hinged([vec2(1.7, 0.3), vec2(1.6, 0.2)])
.styled([vec4(1.1, 0.05, 0.0, 0.0), vec4(0.2, 0.9, 0.0, 0.5)])
.framed([1.15, 0.45])
.edged(6)
.skinned(vec4(3.0, 0.9, 1.8, 0.25))
.flying(vec4(0.35, 0.25, 0.30, 1.30))
.marked(vec4(6.00, 3.00, 0.70, 0.42), vec3(0.26, 0.22, 0.17));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        speed: 45.0,
        flocks: false,
        ..super::super::PHYSIOLOGY
    };
}
