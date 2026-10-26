use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid, prepare_pose, ribbons, rig_ribbons, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail,
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
    size: 0.045,
    maturity_days: 60.0,
    lifespan_days: 900.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 40.4;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Lucerna undulata",
    "Shimmerays",
    [
        "Shimmerays are broad, flat filter feeders whose slow, rippling wingbeats carry a wide scoop of a mouth through currents of algae and drifting organic particles.",
        "A ring of small glowing lures round the scoop draws Sparkies close enough to be swept in with the feeding stream and caught on the comb plates inside.",
        "Long luminous ribbons stream from beneath the body, sampling a deeper volume of air for particles and marking each animal's place in the swarm.",
        "They drift along the sunset band, sinking toward the upper edge of the Veilands to feed and rising again once fed; concentrated aur remains dangerous to their tissues.",
        "Seen from afar, their coordinated vertical movement forms long shifting ribbons of light over the lower Gardens, and their eggs drift up from it on crowns of fine threads.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.5, 5.0, 0);
    c.opaque_bounds = vec3(6.0, 6.0, 22.0).extend(0.0);
    rig_ribbons(c, 3, vec3(0.0, -0.4, 2.6), 1.1, 12.0);
}

/// A living trawl: a broad, flat body like a disc, rippling as it goes.
///
/// It opens in front into a wide slot of a mouth combed inside with plates that catch what the
/// current carries in. Round the scoop's lips a ring of small glowing lures draws Sparkies into it,
/// and from beneath the body three luminous ribbons stream far behind.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let ripple = (p.x.abs() * 0.8 - c.animation.z * 1.5).sin() * 0.12 * p.x.abs();
    let rippled = p - vec3(0.0, ripple, 0.0);
    let disc = rounded(rippled, vec3(3.1, 0.6, 3.6));
    let scoop = rounded(rippled - vec3(0.0, -0.05, -3.5), vec3(1.6, 0.2, 0.9));
    let body = smooth_min(disc.max(-scoop), tail(p, c), 0.4);
    let mut hit = Surface::new(body, 0);
    // Comb plates across the back of the scoop.
    let comb_x = (rippled.x / 0.22).round() * 0.22;
    if comb_x.abs() < 1.5 {
        let comb = vec3(rippled.x - comb_x, rippled.y + 0.05, rippled.z + 2.75);
        hit.join(rounded(comb, vec3(0.02, 0.18, 0.2)), 7);
    }
    // The lures, along both lips.
    let lure_x = ((rippled.x / 0.5).round() * 0.5).clamp(-1.5, 1.5);
    for lip in 0..2 {
        let height = if lip == 0 { 0.32 } else { -0.4 };
        let lure = vec3(lure_x, height, -3.45 + (lure_x * 0.6).powi(2) * 0.3);
        hit.join((rippled - lure).length() - 0.1, 9);
    }
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        ellipsoid(side - vec3(1.1, -0.45, -2.9), vec3(0.2, 0.16, 0.2)),
        3,
    );
    hit.join(ribbons(p, c, 3), 9);
    hit
}

/// The lure gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const LURE: Organ = Organ::new("lure gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.15, -2.55), vec3(0.08, 0.08, 0.08)),
    site(PUMP, vec3(0.00, -0.15, -1.38), vec3(0.40, 0.13, 0.40)),
    site(CROP, vec3(0.00, -0.15, 0.20), vec3(0.31, 0.21, 0.31)),
    pair(BREATHER, vec3(1.99, 0.29, -1.18), vec3(0.50, 0.16, 0.50)),
    site(FLOAT, vec3(0.00, 0.44, 0.20), vec3(0.11, 0.11, 0.11)),
    pair(
        FLIGHT_MUSCLE,
        vec3(1.42, 0.29, -0.39),
        vec3(0.42, 0.13, 0.42),
    ),
    site(CISTERN, vec3(0.00, -0.15, 1.77), vec3(0.62, 0.21, 0.68)),
    site(BROOD, vec3(0.00, 0.15, 2.55), vec3(0.62, 0.23, 0.71)),
    pair(LURE, vec3(0.85, 0.29, -2.75), vec3(0.09, 0.09, 0.09)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.16, 0.26, 0.41, 0.0),
    vec3(0.22, 0.82, 0.89),
    [vec4(7.5, 3.8, -1.4, 0.3), vec4(5.5, 3.0, 1.8, 0.3)],
)
.hinged([vec2(2.7, 0.0), vec2(2.5, 0.0)])
.framed([0.4, 0.4])
.styled([vec4(0.6, 0.8, 0.0, 0.3), vec4(0.6, 0.8, 0.0, 0.3)])
.skinned(vec4(5.0, 1.0, 1.2, -0.25))
.flying(vec4(0.45, 0.30, 0.60, 1.60))
.marked(vec4(5.00, 4.00, 0.70, 0.35), vec3(0.85, 0.35, 0.90));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Brood, Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        brood: Brood::Drifting,
        ..super::super::PHYSIOLOGY
    };
}
