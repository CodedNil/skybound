use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    plantae::{count, frond, gene, rotate, speckle},
    smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.36,
    maturity_days: 7300.0,
    lifespan_days: 365_000.0,
    adult_mass: 60.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 32.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Susurrus tubulatus",
    "Whisperwood",
    [
        "Whisperwood rises around a broad-based hollow trunk whose ridged walls guide the updraft through the body.",
        "Tubular branches end in apertures that whistle at different pitches, surrounding delicate fan leaves with an audible record of the passing air.",
        "Glowgnomes occupy sheltered wood cavities and receive sap while providing light and collective pulling force.",
        "Six vertical sap veins carry pheromone requests to groups on different sides, allowing gradual movement toward favourable humidity or away from danger.",
        "The partnership preserves lift and living support without roots, making an old tree a navigable refuge that travels with its inhabitants.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(15.0, 22.0, 15.0).extend(0.0);
}

/// The foot of the trunk.
const FOOT: f32 = -10.0;

/// The trunk's outer radius at a height, before its ridges: broad at the flared base and
/// narrowing upwards.
fn girth(c: &CreatureInstance, y: f32, tall: f32) -> f32 {
    let flare = 0.4 + gene(c, 2.0) * 1.4;
    2.2 * (1.0 - 0.3 * ((y - FOOT) / (tall - FOOT)).clamp(0.0, 1.0))
        + 2.2 * flare * (-(y - FOOT) / 2.5).exp()
}

/// A whisperwood.
///
/// A hollow trunk, broad and flared at its base, whose six ridged walls carry the sap veins and
/// guide the updraft up through it; tubular branches leave it and end in flared mouths that
/// whistle at different pitches, ringed by delicate fan leaves; and in sheltered cavities in the
/// wood the glowgnomes live. Trees grow as their histories made them: tall or squat, flared at the
/// foot or barely, upright or leaning, with a few great whistling branches or many small ones, and
/// more cavities the older the colony that lives in them.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let tall = 8.0 + gene(c, 1.0) * 10.0;
    let lean = (gene(c, 3.0) - 0.5) * 0.5;
    let bearing = gene(c, 4.0) * TAU;
    let tilt = vec2(bearing.cos(), bearing.sin()) * lean * (p.y - FOOT).max(0.0) * 0.08;
    let leaning = vec3(p.x - tilt.x, p.y, p.z - tilt.y);
    let angle = leaning.z.atan2(leaning.x);
    let radius = girth(c, leaning.y, tall);
    // Six ridges over the sap veins, deep or shallow.
    let ridges = (angle * 6.0).cos().abs().powf(3.0) * (0.15 + gene(c, 5.0) * 0.35);
    let around = leaning.xz().length();
    let outer = around - radius - ridges;
    // Solid at the flared foot; hollow above it, walls about half a metre thick.
    let bore =
        around - (2.2 * (1.0 - 0.3 * ((leaning.y - FOOT) / (tall - FOOT)).clamp(0.0, 1.0)) - 0.6);
    let wall = outer.max(-bore.max(FOOT + 1.5 - leaning.y));
    let trunk = wall.max(leaning.y - tall).max(FOOT - leaning.y) * 0.8;
    let mut shape = Surface::new(trunk, 0);
    // The whistling branches.
    let branches = count(c, 6.0, 3, 8);
    let reach = (11.0 - branches as f32 * 0.8) * (0.8 + gene(c, 7.0) * 0.4);
    for i in 0..branches {
        let f = i as f32;
        let turn = f * 2.39996 + gene(c, 8.0) * TAU;
        let height = FOOT + 4.0 + (tall - FOOT - 4.0) * (0.3 + (f + 0.5) / branches as f32 * 0.7);
        let out = vec3(turn.cos(), 0.0, turn.sin());
        let root = vec3(tilt.x, height, tilt.y) + out * girth(c, height, tall) * 0.8;
        let long = reach * (0.6 + speckle(vec2(f, 1.0), 1.0) * 0.6);
        let rise = 0.3 + speckle(vec2(f, 2.0), 1.0) * 0.8;
        let elbow = root + out * long * 0.6 + vec3(0.0, long * rise * 0.3, 0.0);
        let mouth = elbow + out * long * 0.4 + vec3(0.0, long * rise * 0.5, 0.0);
        if (p - elbow).length() > long + 4.0 {
            continue;
        }
        let pipe = 0.35 + speckle(vec2(f, 3.0), 1.0) * 0.35;
        // One smooth curve from the trunk to the mouth, drawn in pieces that flow into one
        // another, thick where it leaves the wood and narrowing before it flares.
        let curve = |t: f32| root.lerp(elbow, t).lerp(elbow.lerp(mouth, t), t);
        let mut tube = 1e6f32;
        let mut previous = root;
        for piece in 1..=5 {
            let t = piece as f32 / 5.0;
            let next = curve(t);
            let width = pipe * (1.6 - t * 0.7);
            tube = smooth_min(
                tube,
                tapered_tail(p, previous, next, width, width - pipe * 0.14),
                pipe * 0.6,
            );
            previous = next;
        }
        let axis = (mouth - curve(0.8)).normalize();
        let hollow = tapered_tail(p, curve(0.6), mouth + axis * 0.2, pipe * 0.55, pipe * 0.6);
        // The flared mouth that whistles, rolling out of the pipe.
        let along = (p - mouth).dot(axis);
        let off = (p - mouth - axis * along).length();
        let flare = pipe * (0.9 + ((along + 1.0) / 1.0).clamp(0.0, 1.0).powi(2) * 1.5);
        let bell = ((off - flare).abs() - 0.09).max(along).max(-along - 1.0) * 0.8;
        // Grown out of the trunk, its root swelling into the wood.
        shape.blend(
            smooth_min(tube, bell, pipe * 0.4).max(-hollow),
            0,
            pipe * 1.5,
        );
        // Fan leaves round the mouth.
        for leaf in 0..count(c, 9.0, 0, 3) {
            let spin = leaf as f32 * 2.1 + f;
            let local = rotate(p - mouth + axis * 0.6, turn + spin);
            shape.join(frond(local, 2.2 + gene(c, 10.0) * 1.5, 0.9, 0.6, 2.0), 1);
        }
    }
    // Cavities in the wood, lit by the glowgnomes living in them.
    for i in 0..count(c, 11.0, 1, 5) {
        let f = i as f32;
        let turn = f * 2.7 + 0.6;
        let height = FOOT + 3.0 + f * 2.6;
        let at = vec3(turn.cos(), 0.0, turn.sin()) * (girth(c, height, tall) + 0.1)
            + vec3(tilt.x, height, tilt.y);
        let hole = (p - at).length() - 0.75;
        shape.distance = shape.distance.max(-hole);
        shape.join((p - at).length() - 0.35, 2);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.60, -1.90), vec3(0.20, 0.20, 0.20)),
    site(BLADDER, vec3(0.00, 3.00, -1.70), vec3(0.15, 0.15, 0.15)),
    pair(FROND, vec3(1.62, 1.18, -0.70), vec3(0.13, 0.13, 0.13)),
    site(ROOT, vec3(0.00, -9.50, -1.90), vec3(0.30, 0.30, 0.30)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.15, 0.44, 0.3), vec3(0.37, 0.85, 0.56));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
