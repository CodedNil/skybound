use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{
        count, gene, pick, rotate, speckle,
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
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.29,
    maturity_days: 35.0,
    lifespan_days: 700.0,
    adult_mass: 0.5,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 11.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Polliniflora nectarifera",
    "Blooms",
    [
        "Blooms hold layered petals around reservoirs of sweet, aur-rich nectar on flexible stems.",
        "A buoyant body maintains the flower in rising air while the stem yields beneath an approaching Misting.",
        "Pollen brushes the visitor during feeding and travels onward when the animal leaves for another drifting plant.",
        "Hygroscopic surfaces supply water, and vascular channels deliver the minerals needed to renew petals and nectar.",
        "Their healing tissues are valued by Næ, so careful gathering preserves a living resource that also feeds the surrounding pollinator community.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

/// The middle of the float the stems rise from.
const FLOAT: Vec3 = vec3(0.0, -1.4, 0.0);

/// A bloom.
///
/// A buoyant float holding up one flower or a spray of them on flexible stems that give under a
/// Misting's weight, with fine roots trailing below. Blooms differ far more than their giant kin:
/// one flower or five, held up on long stems or nodding on short ones, as open stars, cups,
/// frilled rosettes or hanging bells with trailing petals, over a round float, a pear-shaped one
/// or a pair.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(1e6, 3);
    match pick(c, 1.0, 3) {
        0 => shape.join(ellipsoid(p - FLOAT, vec3(1.2, 1.0, 1.2)), 3),
        1 => shape.join(
            ellipsoid(p - FLOAT - vec3(0.0, -0.3, 0.0), vec3(0.9, 1.4, 0.9)),
            3,
        ),
        _ => {
            let twin = vec3((p.x - FLOAT.x).abs() - 0.6, p.y - FLOAT.y, p.z - FLOAT.z);
            shape.join(ellipsoid(twin, vec3(0.85, 0.8, 0.85)), 3);
        }
    }
    let heads = count(c, 2.0, 1, 5);
    let reach = 1.5 + gene(c, 3.0) * 2.5;
    let nod = gene(c, 4.0);
    let form = pick(c, 5.0, 4);
    let bell = form == 3;
    let shape_of = Flower {
        petals: count(c, 6.0, 4, 9),
        whorls: count(c, 7.0, 1, 2),
        form,
        length: (2.6 - heads as f32 * 0.3) * (0.8 + gene(c, 8.0) * 0.5),
        width: 0.7 + gene(c, 9.0) * 0.7,
        open: if bell {
            -0.6
        } else {
            0.1 + gene(c, 10.0) * 0.9
        },
        stamens: count(c, 11.0, 3, 8),
    };
    for i in 0..heads {
        let f = i as f32;
        let bearing = f * TAU / heads as f32 + gene(c, 12.0) * TAU;
        let out = if heads == 1 {
            Vec3::ZERO
        } else {
            vec3(bearing.cos(), 0.0, bearing.sin()) * (0.8 + speckle(vec2(f, 1.0), 1.0))
        };
        let sway = (c.animation.z * 0.9 + f).sin() * 0.15;
        let rise = reach * (0.7 + speckle(vec2(f, 2.0), 1.0) * 0.5);
        let head = FLOAT + out * (1.0 + nod) + vec3(sway, rise + 0.8, 0.0);
        let bend = FLOAT + out * 0.4 + vec3(0.0, rise * 0.6 + 0.6, 0.0);
        shape.join(
            tapered_tail(p, FLOAT, bend, 0.16, 0.12).min(tapered_tail(p, bend, head, 0.12, 0.08)),
            0,
        );
        // Bells hang their throats down and out; the rest face up and away from the float.
        let lean = if bell {
            2.2
        } else if heads > 1 {
            nod * 0.9
        } else {
            0.0
        };
        let q = rotate(p - head, bearing);
        let (sine, cosine) = lean.sin_cos();
        let q = vec3(q.x * cosine - q.y * sine, q.x * sine + q.y * cosine, q.z);
        let one = flower(q, shape_of, c.animation.z);
        if one.distance < shape.distance {
            shape = one;
        }
    }
    for i in 0..count(c, 13.0, 3, 8) {
        let f = i as f32;
        let bearing = f * 2.39996;
        let sway = (c.animation.z * 0.5 + f).sin() * 0.2;
        let end = FLOAT
            + vec3(
                bearing.cos() * 0.6 + sway,
                -2.0 - gene(c, 14.0 + f) * 2.5,
                bearing.sin() * 0.6,
            );
        shape.join(
            tapered_tail(p, FLOAT - vec3(0.0, 0.6, 0.0), end, 0.06, 0.015),
            0,
        );
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.10, 0.00), vec3(0.25, 0.25, 0.25)),
    site(BLADDER, vec3(0.00, -1.50, 0.30), vec3(0.30, 0.30, 0.30)),
    pair(FROND, vec3(0.35, -1.40, 0.00), vec3(0.15, 0.15, 0.15)),
    site(ROOT, vec3(0.00, -1.85, 0.00), vec3(0.20, 0.15, 0.20)),
    site(BLOOM, vec3(0.00, -1.30, -0.35), vec3(0.20, 0.20, 0.20)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.82, 0.16, 0.43), vec3(1.0, 0.68, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
