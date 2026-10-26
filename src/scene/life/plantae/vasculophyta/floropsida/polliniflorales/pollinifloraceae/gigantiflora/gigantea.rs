use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{
        count, frond, gene, pick, rotate,
        vasculophyta::floropsida::polliniflorales::pollinifloraceae::{Flower, flower},
    },
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::{
        physiology::{BLADDER, FROND, ROOT, STEM},
        vasculophyta::floropsida::BLOOM,
    },
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.75,
    maturity_days: 400.0,
    lifespan_days: 18000.0,
    adult_mass: 12.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 30.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Gigantiflora gigantea",
    "Giant Blooms",
    [
        "Giant Blooms raise luminous flowers that approach the height of small trees above buoyant supporting bodies.",
        "Broad petals surround deep nectar reservoirs supplied through flexible vascular tissues.",
        "Their scale rewards large visitors whose energy needs cannot be met efficiently by ordinary Blooms.",
        "Slow, steady currents in fertile aur-rich Gardens support the heavy flowering structures and reduce damage during feeding.",
        "A rare successful visit can carry pollen across a large gap, making dependable attraction more useful than producing many tiny flowers.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(16.0, 22.0, 16.0).extend(0.0);
}

/// The middle of the buoyant body the whole plant rises from.
const BODY: Vec3 = vec3(0.0, -10.0, 0.0);

/// Where along the stem a point at `t` lies, from the buoyant body to the flower: rising, and
/// bowing over towards its lean as it nears the flower's weight.
fn stem(c: &CreatureInstance, t: f32) -> Vec3 {
    let tall = 10.0 + gene(c, 1.0) * 9.0;
    let bow = gene(c, 2.0) * 0.9;
    let bearing = gene(c, 3.0) * TAU;
    let sway = (c.animation.z * 0.4).sin() * 0.25 * t;
    BODY + vec3(0.0, 2.0, 0.0)
        + vec3(bearing.cos(), 0.0, bearing.sin()) * (t * t * tall * bow * 0.5 + sway)
        + vec3(0.0, t * tall * (1.0 - bow * 0.2 * t), 0.0)
}

/// A giant bloom.
///
/// A great luminous flower raised to the height of a small tree on a vascular stem above a
/// buoyant body, with deep nectar stored in the body and drawn up the stem to the flower's
/// glowing cup. Each grows its own: a tall stem or a short one, upright or bowed over so the
/// flower faces out, leafless or leafy, five broad petals or a dozen narrow ones, in one whorl or
/// three, pointed, rounded, frilled or trailing; and its body is one great float, a cluster of
/// them, or a long gourd.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(1e6, 3);
    match pick(c, 4.0, 3) {
        0 => shape.join(ellipsoid(p - BODY, vec3(3.4, 2.8, 3.4)), 3),
        1 => {
            shape.join(ellipsoid(p - BODY, vec3(2.4, 2.2, 2.4)), 3);
            for i in 0..3 {
                let angle = i as f32 * TAU / 3.0 + gene(c, 5.0);
                let at = BODY + vec3(angle.cos() * 2.4, -0.6, angle.sin() * 2.4);
                shape.join(ellipsoid(p - at, vec3(1.9, 1.7, 1.9)), 3);
            }
        }
        _ => shape.join(
            ellipsoid(p - BODY - vec3(0.0, -1.5, 0.0), vec3(2.3, 4.2, 2.3)),
            3,
        ),
    }
    // The stem, drawn in pieces along its curve.
    let thick = 0.45 + gene(c, 6.0) * 0.45;
    let mut previous = stem(c, 0.0);
    for k in 1..=6 {
        let t = k as f32 / 6.0;
        let next = stem(c, t);
        shape.join(
            tapered_tail(
                p,
                previous,
                next,
                thick * (1.1 - t * 0.4),
                thick * (1.0 - t * 0.4),
            ),
            0,
        );
        previous = next;
    }
    for i in 0..count(c, 7.0, 0, 6) {
        let t = 0.15 + i as f32 * 0.12;
        let at = stem(c, t);
        let angle = i as f32 * 2.4 + gene(c, 8.0);
        shape.join(
            frond(
                rotate(p - at, angle),
                4.0 + gene(c, 9.0) * 3.0,
                1.6,
                1.0,
                3.0,
            ),
            1,
        );
    }
    // The flower, facing along the stem's last stretch.
    let head = stem(c, 1.0);
    let axis = (head - stem(c, 0.85)).normalize();
    let bearing = axis.z.atan2(axis.x);
    let tilt = axis.y.clamp(-1.0, 1.0).acos();
    let q = rotate(p - head, bearing);
    let (sine, cosine) = tilt.sin_cos();
    let q = vec3(q.x * cosine - q.y * sine, q.x * sine + q.y * cosine, q.z);
    let open = flower(
        q,
        Flower {
            petals: count(c, 10.0, 5, 11),
            whorls: count(c, 11.0, 1, 3),
            form: pick(c, 12.0, 4),
            length: 6.0 + gene(c, 13.0) * 4.0,
            width: 1.6 + gene(c, 14.0) * 1.8,
            open: 0.15 + gene(c, 15.0) * 0.7,
            stamens: count(c, 16.0, 0, 9),
        },
        c.animation.z,
    );
    if open.distance < shape.distance {
        shape = open;
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -8.40, 0.00), vec3(0.60, 0.60, 0.60)),
    site(BLADDER, vec3(0.00, -10.20, 1.00), vec3(0.90, 0.90, 0.90)),
    pair(FROND, vec3(0.90, -9.60, 0.00), vec3(0.60, 0.60, 0.60)),
    site(ROOT, vec3(0.00, -11.40, -0.60), vec3(0.70, 0.60, 0.70)),
    site(BLOOM, vec3(0.00, -9.80, -1.20), vec3(0.80, 0.80, 0.80)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.48, 0.2, 0.77), vec3(0.28, 0.85, 1.0));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
