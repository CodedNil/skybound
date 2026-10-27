use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::Biology,
    eyespots, feathers,
    genome::Gene,
    prepare_pose, rounded, smooth_min,
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
    organs::{Viscera, pair, site},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.36,
    maturity_days: 1460.0,
    lifespan_days: 40000.0,
    adult_mass: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 49.8;

/// How wide its funnel flares, how large the gas store over its back, and how many comb plates line
/// the funnel.
///
/// A wide funnel strains more of a stream, a large store carries it further between streams, and
/// close-set combs catch finer algae.
pub const FLARE: Gene = Gene(38);
pub const STORE: Gene = Gene(39);
pub const COMBS: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Herbivolus migratorius",
    "Grazers",
    [
        "Grazers are large herbivores without heads: the front of each has become a great flared funnel lined with comb plates, held open behind broad feathered wings.",
        "They fly through dense algae concentrations with the funnel open, straining food from the air while soft mouth tentacles about the rim scrape the combs clean.",
        "Herds space themselves across a rising food stream so that neighbours do not continually feed in one another's depleted wake.",
        "Repeated journeys to the Mineral Fields replenish minerals absent from clearer feeding air.",
        "Their migration routes link productive Gardens to dusty regions, carrying predators, cleaners and attached parasites between the same communities.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 9.0, 14.0, 0);
    c.opaque_bounds = vec3(7.0, 7.0, 24.0).extend(0.0);
    rig_tentacles(c, RIM, 2.3, ORAL_PAIRS);
}
/// The low point of the funnel's rim, where the tentacles hang.
const RIM: Vec3 = vec3(0.0, -3.4, -14.0);

/// A living trawl with no head at all.
///
/// Its front has become one great flared funnel, lined inside with comb plates, held open as it
/// flies through an algae stream so the food is strained out of the air it swallows; the three
/// pairs of tentacles hang about the rim to scrape the combs clean. Behind the funnel lies a barrel
/// of a body, swollen with a gas store over the back that cuts the cost of carrying it between
/// streams, and its eyes run in rows along the lower flanks, watching the neighbours a herd keeps
/// its spacing by. Each animal's funnel and store are the size its genes make them.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let flare = 0.85 + c.gene(FLARE) * 0.35;
    let barrel = rounded(p - vec3(0.0, 0.0, 1.0), vec3(3.4, 3.1, 10.0));
    let store = rounded(
        p - vec3(0.0, 2.5, 0.0),
        vec3(2.4, 1.4, 4.5) * (0.8 + c.gene(STORE) * 0.4),
    );
    let mut body = smooth_min(smooth_min(barrel, store, 1.4), tail(p, c), 0.5);
    let q = p - vec3(0.0, 0.3, -8.0);
    let depth = -q.z;
    let t = (depth / 6.0).clamp(0.0, 1.0);
    let radius = (1.6 + 3.0 * t) * flare;
    let radial = vec2(q.x, q.y).length();
    let funnel = ((radial - radius).abs() - 0.35 * (1.0 - t * 0.5))
        .max(-depth)
        .max(depth - 6.0)
        * 0.7;
    body = smooth_min(body, funnel, 0.8);
    let mut hit = Surface::new(body, 0);
    let around = q.y.atan2(q.x);
    let sector = TAU / (18.0 + (c.gene(COMBS) * 8.0).floor());
    let turn = (around / sector).round() * sector;
    let across = radial * (around - turn).sin();
    let comb = (across.abs() - 0.05)
        .max(radial - radius + 0.2)
        .max(radius - 1.1 - radial)
        .max(3.5 - depth)
        .max(depth - 5.8)
        * 0.7;
    hit.join(comb, 7);
    hit.join(feathers(p, c, 4.0), 1);
    hit.socket(eyespots(p, vec3(2.9, -1.5, -6.5), 1.4, 4, 0.24), 0.24);
    hit.blend(oral_tentacles(p, c, RIM, 2.3, ORAL_PAIRS), 8, 0.23);
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.62, -6.21), vec3(0.61, 0.57, 1.32)),
    site(PUMP, vec3(0.00, -0.09, -2.36), vec3(1.36, 1.36, 1.99)),
    site(CROP, vec3(0.00, -0.09, 1.93), vec3(1.42, 1.42, 3.83)),
    pair(BREATHER, vec3(2.26, 0.89, -3.00), vec3(0.44, 0.44, 0.44)),
    site(FLOAT, vec3(0.00, 3.30, -1.07), vec3(0.45, 0.45, 0.45)),
    pair(
        FLIGHT_MUSCLE,
        vec3(2.26, 1.61, -1.29),
        vec3(0.37, 0.37, 0.37),
    ),
    site(CISTERN, vec3(0.00, -0.09, 6.64), vec3(0.91, 0.91, 1.99)),
    site(BROOD, vec3(0.00, 0.45, 9.21), vec3(0.66, 0.62, 1.00)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.68, 0.54, 0.31, 0.0),
    vec3(0.87, 0.75, 0.54),
    [vec4(20.0, 6.0, -1.5, 0.98), vec4(14.0, 4.0, 4.0, 0.98)],
)
.hinged([vec2(3.3, 1.2), vec2(3.0, 1.0)])
.styled([vec4(0.5, 0.8, 5.0, 0.0), vec4(0.6, 0.7, 0.0, 0.0)])
.skinned(vec4(1.0, 1.3, 0.8, 0.05))
.flying(vec4(0.30, 0.45, 0.40, 1.20))
.marked(vec4(1.00, 9.00, 0.35, 0.42), vec3(0.35, 0.27, 0.16))
.brooding(vec4(3.0, 6.0, 1.8, 1.3));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Sunlit,
        speed: 12.0,
        ..super::super::PHYSIOLOGY
    };
}
