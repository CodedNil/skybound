use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyespots, feathers, prepare_pose, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1460.0,
    lifespan_days: 40000.0,
    speed: 12.0,
    adult_mass: 35.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 49.8;

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
    rig_tentacles(c, RIM, 2.3, BIOLOGY.oral_pairs);
}
/// The low point of the funnel's rim, where the tentacles hang.
const RIM: Vec3 = vec3(0.0, -3.4, -14.0);

/// A living trawl with no head at all.
///
/// Its front has become one great flared funnel, lined inside with comb plates,
/// held open as it flies through an algae stream so the food is strained out of
/// the air it swallows; the three pairs of tentacles hang about the rim to
/// scrape the combs clean. Behind the funnel lies a barrel of a body, swollen
/// with a gas store over the back that cuts the cost of carrying it between
/// streams, and its eyes run in rows along the lower flanks, watching the
/// neighbours a herd keeps its spacing by. Each animal's funnel and store are
/// its own size.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let flare = 0.85 + c.identity.y * 0.35;
    let barrel = rounded(p - vec3(0.0, 0.0, 1.0), vec3(3.4, 3.1, 10.0));
    let store = rounded(
        p - vec3(0.0, 2.5, 0.0),
        vec3(2.4, 1.4, 4.5) * (0.8 + c.identity.z * 0.4),
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
    let sector = TAU / 22.0;
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
    hit.join(eyespots(p, vec3(2.9, -1.5, -6.5), 1.4, 4, 0.24), 3);
    hit.join(oral_tentacles(p, c, RIM, 2.3, BIOLOGY.oral_pairs), 8);
    hit
}

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
