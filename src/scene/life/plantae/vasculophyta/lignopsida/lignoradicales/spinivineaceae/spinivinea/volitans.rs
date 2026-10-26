use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, gene, leaf, pick, rotate, speckle},
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, Physiology, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.85,
    maturity_days: 1000.0,
    lifespan_days: 70000.0,
    adult_mass: 8.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 13.0;

/// Brambles grow in linked chains of tangles, each spore taking hold on the chain it fell from.
#[cfg(not(target_arch = "spirv"))]
pub const PHYSIOLOGY: Physiology = Physiology {
    stand: 7,
    ..super::PHYSIOLOGY
};

taxon!(
    Species,
    "Spinivinea volitans",
    "Brambles",
    [
        "Brambles form dense spheres and linked chains of woody vines supported by large lotus-shaped leaves.",
        "Their thorns carry a mild auric charge that discourages most large grazers from pressing into the tangle.",
        "Small animals use interior gaps as shelter, while Driftmunchers work carefully through bark and spines on the outside.",
        "Flexible connections let a chain bend around uneven airflow rather than behave as a rigid floating wall.",
        "Spores establish new tangles as older chains separate, continually changing the passages and refuges inside a Bramble community.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(8.0, 8.0, 9.0).extend(0.0);
}

/// One tangle of the chain, `p` relative to its middle: woody vines wound round and round it in
/// rings at every angle, set with charged thorns.
fn tangle(p: Vec3, c: &CreatureInstance, reach: f32, salt: f32) -> Surface {
    let mut shape = Surface::new(1e6, 0);
    let vines = count(c, 1.0, 5, 10);
    let thorns = 10.0 + gene(c, 2.0) * 22.0;
    let spike = 0.2 + gene(c, 3.0) * 0.7;
    let thick = 0.12 + gene(c, 4.0) * 0.12;
    for i in 0..vines {
        let f = i as f32;
        let yaw = speckle(vec2(f, salt), 1.0) * TAU;
        let roll = speckle(vec2(f, salt), 2.0) * 3.1;
        let q = rotate(p, yaw);
        let (sine, cosine) = roll.sin_cos();
        let q = vec3(q.x, q.y * cosine - q.z * sine, q.y * sine + q.z * cosine);
        let radius = reach * (0.75 + speckle(vec2(f, salt), 3.0) * 0.3);
        let around = vec2(q.x, q.z).length();
        let vine = vec2(around - radius, q.y).length() - thick;
        if vine > spike + 0.5 {
            shape.join(vine, 0);
            continue;
        }
        shape.join(vine, 0);
        let sector = TAU / thorns;
        let angle = q.z.atan2(q.x);
        let at = (angle / sector).round() * sector;
        let out = vec3(at.cos(), 0.0, at.sin());
        let base = out * radius;
        shape.join(
            tapered_tail(
                q,
                base,
                base + (out + vec3(0.0, 0.4, 0.0)) * spike,
                thick * 0.8,
                0.01,
            ),
            0,
        );
    }
    shape
}

/// A bramble.
///
/// Dense spheres of woody vines, wound so thickly that small animals shelter in the gaps, set
/// with thorns carrying a mild charge that keeps the large grazers off, and buoyed by great
/// lotus-shaped leaves spread around them. A plant grows as one tangle, as a chain of tangles
/// joined by bending vines, or as a ring; its vines few or many, thorns short and sparse or long
/// and thick-set, leaves few and huge or many and small.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p, vec3(0.8, 0.8, 0.8)), 0);
    let form = pick(c, 5.0, 3);
    let links = if form == 1 { count(c, 6.0, 2, 3) } else { 1 };
    let reach = (if form == 1 { 2.6 } else { 3.4 }) * (0.85 + gene(c, 7.0) * 0.25);
    for link in 0..links {
        let centre = vec3(
            0.0,
            0.0,
            link as f32 * reach * 1.7 - (links - 1) as f32 * reach * 0.85,
        );
        let local = p - centre;
        if form == 2 {
            // A wreath of small tangles wound round a hollow, each joined to the next.
            let knots = 7.0;
            let sector = TAU / knots;
            let angle = local.z.atan2(local.x);
            let nearest = (angle / sector).floor();
            for side in 0..2 {
                let at = (nearest + side as f32) * sector;
                let knot = vec3(at.cos(), 0.0, at.sin()) * reach;
                shape.join(capsule(p, Vec3::ZERO, knot, 0.18), 0);
                let one = tangle(local - knot, c, reach * 0.42, nearest + side as f32);
                if one.distance < shape.distance {
                    shape = one;
                }
            }
        } else {
            let one = tangle(local, c, reach, link as f32);
            if one.distance < shape.distance {
                shape = one;
            }
        }
        shape.join(capsule(p, Vec3::ZERO, centre, 0.22), 0);
    }
    // The lotus leaves that hold it up.
    let leaves = count(c, 8.0, 3, 8);
    let broad = (4.0 - leaves as f32 * 0.3) * (0.8 + gene(c, 9.0) * 0.4);
    for i in 0..leaves {
        let f = i as f32;
        let bearing = f * TAU / leaves as f32 + gene(c, 10.0);
        let height = (speckle(vec2(f, 7.0), 1.0) - 0.4) * reach;
        let out = vec3(bearing.cos(), 0.0, bearing.sin());
        let stalk = out * reach * 0.9 + vec3(0.0, height, 0.0);
        let rest = out * (reach + broad * 0.3) + vec3(0.0, height + 0.5, 0.0);
        shape.join(capsule(p, stalk, rest, 0.1), 0);
        let q = rotate(p - rest, bearing - TAU / 4.0) + vec3(0.0, 0.0, broad);
        shape.join(leaf(q, broad * 2.0, broad, broad * 0.25), 1);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.00, 0.00), vec3(0.30, 0.30, 0.30)),
    site(BLADDER, vec3(0.00, 0.45, 0.00), vec3(0.15, 0.15, 0.15)),
    pair(FROND, vec3(0.45, 0.00, 0.00), vec3(0.15, 0.15, 0.15)),
    site(ROOT, vec3(0.00, -0.45, 0.00), vec3(0.15, 0.15, 0.15)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.16, 0.35, 0.1), vec3(0.4, 0.6, 0.14));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
