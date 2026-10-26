use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, gene, pick, rotate, speckle},
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.85,
    maturity_days: 1200.0,
    lifespan_days: 60000.0,
    adult_mass: 8.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 20.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Tempestivaga fulgurivora",
    "Stormchasers",
    [
        "Stormchasers carry broad upright sails around a slender woody body strengthened for turbulent air.",
        "Electrical changes in an approaching storm trigger adjustments that expose collecting veins while altering the plant's drift.",
        "Captured discharge passes into protected storage tissues, providing energy between charging opportunities.",
        "After a storm passes, changed sail orientation carries the plant onward rather than leaving it permanently in the most violent air.",
        "Their gathering is a warning of electrical activity, especially where humid geyser-fed currents meet stronger poleward winds.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(11.0, 13.0, 11.0).extend(0.0);
}

/// The middle of the float at the foot of the body.
const FLOAT: Vec3 = vec3(0.0, -3.2, 0.0);

/// A sail, `p` in the sail's own frame with its luff along the y axis and its belly towards +x,
/// of one of three cuts: a tall, battened triangle, a rounded kite, or a sail split into two
/// streaming tongues. The collecting veins run along its battens.
fn sail(p: Vec3, form: u32, tall: f32, wide: f32, belly: f32) -> Surface {
    let t = (p.y / tall).clamp(0.0, 1.0);
    let chord = match form {
        0 => wide * (1.0 - t),
        1 => wide * (t * PI).sin().max(0.0).sqrt(),
        _ => wide * (1.0 - t * 0.6),
    };
    let x = p.x.max(0.0);
    let mut edge = x - chord;
    if form == 2 {
        // The fork, cut from the leech in towards the luff.
        let notch = (p.y - tall * 0.55).abs() - (x / wide) * tall * 0.25;
        edge = edge.max(-notch.max(wide * 0.35 - x));
    }
    let curve = belly * (x / wide.max(0.1)) * (1.0 - x / wide.max(0.1)) * 4.0;
    let cloth = (p.z - curve).abs() - 0.04;
    let mut shape = Surface::new(cloth.max(edge).max(-p.x).max(-p.y).max(p.y - tall) * 0.8, 1);
    let batten = ((p.y / (tall / 7.0)).fract() - 0.5).abs() * (tall / 7.0);
    shape.join(
        (vec2(batten, p.z - curve).length() - 0.06)
            .max(edge)
            .max(-p.x)
            .max(-p.y)
            .max(p.y - tall),
        2,
    );
    shape
}

/// A stormchaser.
///
/// A slender woody body rising from a float, strengthened against turbulent air, carrying broad
/// upright sails all round it; when a storm nears, the plant turns them to bare their collecting
/// veins to the charge, and stores what it catches in glowing nodes along the body. Plants grow
/// two sails or five, battened triangles, rounded kites or forked tongues; tall bodies or short;
/// few stores or many; and some raise lightning spires above the sails, singly or in a crown.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let height = 6.0 + gene(c, 1.0) * 5.0;
    let top = vec3(0.0, FLOAT.y + 1.0 + height, 0.0);
    let mut shape = Surface::new(ellipsoid(p - FLOAT, vec3(1.15, 1.25, 1.15)), 3);
    shape.join(tapered_tail(p, FLOAT, top, 0.42, 0.18), 0);
    for i in 0..count(c, 2.0, 2, 6) {
        let t = (i as f32 + 0.7) / 6.5;
        let at = FLOAT.lerp(top, t);
        shape.join(
            ellipsoid(
                p - at,
                vec3(0.55, 0.42, 0.55) * (0.7 + speckle(vec2(i as f32, 1.0), 1.0) * 0.6),
            ),
            2,
        );
    }
    let sails = count(c, 3.0, 2, 5);
    let form = pick(c, 4.0, 3);
    let tall = height * (0.65 + gene(c, 5.0) * 0.3);
    let wide = (6.0 - sails as f32 * 0.6) * (0.8 + gene(c, 6.0) * 0.4);
    let twist = gene(c, 7.0) * 0.8;
    for i in 0..sails {
        let f = i as f32;
        let bearing = f * TAU / sails as f32 + gene(c, 8.0) * TAU;
        let foot = FLOAT + vec3(0.0, 1.5 + (height - tall) * 0.4, 0.0);
        let local = rotate(p - foot, bearing);
        let local = rotate(local, (local.y / tall).clamp(0.0, 1.0) * twist);
        let one = sail(local, form, tall, wide, 0.4 + gene(c, 9.0) * 0.6);
        if one.distance < shape.distance {
            shape = one;
        }
    }
    match pick(c, 10.0, 3) {
        0 => {}
        1 => shape.join(
            tapered_tail(p, top, top + vec3(0.0, 2.5, 0.0), 0.15, 0.02),
            6,
        ),
        _ => {
            for i in 0..5 {
                let angle = i as f32 * TAU / 5.0;
                let tip = top + vec3(angle.cos() * 1.0, 1.8, angle.sin() * 1.0);
                shape.join(capsule(p, top, tip, 0.06), 6);
            }
        }
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -1.80, 0.00), vec3(0.15, 0.30, 0.15)),
    site(BLADDER, vec3(0.00, -3.20, 0.40), vec3(0.45, 0.45, 0.45)),
    pair(FROND, vec3(0.50, -3.20, 0.00), vec3(0.30, 0.30, 0.30)),
    site(ROOT, vec3(0.00, -4.00, 0.00), vec3(0.30, 0.20, 0.30)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.16, 0.22, 0.46), vec3(0.55, 0.7, 1.0));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        powered: |climate| climate.storm * 8.0,
        ..super::super::PHYSIOLOGY
    };
}
