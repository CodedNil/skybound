use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::Biology,
    ellipsoid, louvres, prepare_pose, rounded, smooth_min,
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
    organs::{Function, Organ, Viscera, pair, site},
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 730.0,
    lifespan_days: 14600.0,
    adult_mass: 5.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

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
        "Small animals drawn up to the lure's apparent feeding signal rise into a short banked stoop, and the iron-hardened leading edges of the forewings, curved to a scythe point, cut across them.",
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
        c.tail[i].y -= t * t * 7.0;
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
    let ruff = ellipsoid(p - vec3(0.0, 0.35, -1.9), vec3(2.1, 0.75, 0.9))
        + (p.x * 9.0).sin() * (p.z * 7.0).sin() * 0.06;
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
    let mut hit = Surface::new(smooth_min(d, whip, 0.5), 0);
    // The lure: a bulb trailing glowing threads.
    let end = c.tail[5].truncate();
    let mut lure = ellipsoid(p - end, vec3(0.35, 0.45, 0.35));
    for thread in 0..3 {
        let f = thread as f32 - 1.0;
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
        ellipsoid(side - vec3(0.6, -0.55, -3.75), vec3(0.55, 0.5, 0.6)),
        3,
    );
    hit.join(oral_tentacles(p, c, MOUTH, 0.8, ORAL_PAIRS), 8);
    hit
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
