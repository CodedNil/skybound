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
    organs::{Viscera, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.25,
    maturity_days: 180.0,
    lifespan_days: 9000.0,
    adult_mass: 0.2,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 10.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Chromacorallum versicolor",
    "Color Coral",
    [
        "Colour Coral spreads branching fans of green, yellow, pink, blue and purple across kelp and Laevensail growth.",
        "Frilled surfaces collect dispersed aur while narrow attachments leave much of the host exposed to passing air.",
        "Flexible junctions let each fan turn independently when a long frond twists in the current.",
        "A faint glow reveals living margins in the dim equatorial skies and attracts small organisms into the sheltered branches.",
        "The colony gains support and transport, while excessive growth can shade or weigh down the plant carrying it.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 6.0, 5.0).extend(0.0);
}

/// Where the colony holds on to its host.
const FOOT: Vec3 = vec3(0.0, -1.4, 0.0);

/// A lattice fan, `p` in the fan's own plane rising from its root: branches spreading from the
/// root and joined by cross-ties into a net that sieves the passing air.
fn lattice(p: Vec3, reach: f32, branches: f32) -> f32 {
    let along = vec2(p.x, p.y).length();
    let angle = p.x.atan2(p.y);
    let spread = 1.1;
    let sector = spread * 2.0 / branches;
    let nearest = ((angle + spread) / sector).round() * sector - spread;
    let ray = ((angle - nearest) * along).abs();
    let ring = ((along / 0.45).fract_gl() - 0.5).abs() * 0.45;
    let net = ray.min(ring) - 0.035;
    let outline = (angle.abs() - spread).max(along - reach);
    // The thickened rim, where the faint glow of the living margin shows.
    let rim = vec2(along - reach, p.z).length() - 0.06 + (angle.abs() - spread).max(0.0) * along;
    net.max(outline).max(p.z.abs() - 0.04).min(rim)
}

/// A colony of colour coral.
///
/// It grips a frond or branch of its host by a narrow foot, leaving the rest exposed to the air,
/// and grows in one of three ways: as lattice fans netting the current, each turning on a flexible
/// joint; as a bushy thicket of forking branches tipped with glowing buds; or as a cluster of
/// upright pipes whose rims glow. Every colony takes one of its lineage's colours.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - FOOT, vec3(0.6, 0.35, 0.6)), 0);
    shape.join(capsule(p, FOOT, vec3(0.0, 0.6, 0.0), 0.28), 0);
    let base = vec3(0.0, 0.6, 0.0);
    match pick(c, 1.0, 3) {
        0 => {
            for i in 0..count(c, 2.0, 1, 3) {
                let f = i as f32;
                let turn = f * 1.9 + gene(c, 3.0) * PI;
                let sway = (c.animation.z * 0.7 + f * 2.0).sin() * 0.15;
                let local = rotate(p - base, turn + sway);
                let tilt = (f - 1.0) * 0.35;
                let q = vec3(
                    local.x * tilt.cos() - local.y * tilt.sin(),
                    local.x * tilt.sin() + local.y * tilt.cos(),
                    local.z,
                );
                let reach = 3.0 + gene(c, 4.0 + f) * 1.8;
                shape.join(lattice(q, reach, 5.0 + gene(c, 5.0) * 6.0), 1);
            }
        }
        1 => {
            let arms = count(c, 6.0, 3, 6);
            for i in 0..arms {
                let f = i as f32;
                let bearing = f * TAU / arms as f32 + gene(c, 7.0) * TAU;
                let out = vec3(bearing.cos(), 0.0, bearing.sin());
                let fork = base + out * 1.2 + vec3(0.0, 1.4 + speckle(vec2(f, 1.0), 1.0), 0.0);
                shape.join(tapered_tail(p, base, fork, 0.16, 0.1), 1);
                for twig in 0..3 {
                    let t = twig as f32 - 1.0;
                    let side = vec3(-out.z, 0.0, out.x) * t * 0.7;
                    let tip = fork
                        + out * 0.9
                        + side
                        + vec3(0.0, 1.0 + speckle(vec2(f, t), 2.0) * 0.8, 0.0);
                    shape.join(tapered_tail(p, fork, tip, 0.1, 0.05), 1);
                    shape.join((p - tip).length() - 0.14, 2);
                }
            }
        }
        _ => {
            for i in 0..count(c, 8.0, 4, 9) {
                let f = i as f32;
                let bearing = f * 2.39996 + gene(c, 9.0) * TAU;
                let out =
                    vec3(bearing.cos(), 0.0, bearing.sin()) * (0.2 + (f * 0.31).fract_gl() * 0.9);
                let tall = 1.4 + speckle(vec2(f, 3.0), 1.0) * 2.2;
                let wide = 0.16 + speckle(vec2(f, 4.0), 1.0) * 0.16;
                let foot = base + out * 0.5;
                let top = foot + out * 0.6 + vec3(0.0, tall, 0.0);
                let tube = tapered_tail(p, foot, top, wide * 0.7, wide);
                let bore = capsule(
                    p,
                    foot + (top - foot) * 0.5,
                    top + (top - foot) * 0.2,
                    wide * 0.65,
                );
                shape.join(tube.max(-bore), 1);
                shape.join(
                    vec2(
                        vec2(p.x - top.x, p.z - top.z).length() - wide * 0.85,
                        p.y - top.y,
                    )
                    .length()
                        - 0.05,
                    2,
                );
            }
        }
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.30, 0.00), vec3(0.15, 0.30, 0.15)),
    site(BLADDER, vec3(0.00, 0.40, 0.00), vec3(0.10, 0.10, 0.10)),
    site(FROND, vec3(0.00, 0.20, 0.10), vec3(0.08, 0.08, 0.08)),
    site(ROOT, vec3(0.00, -1.40, 0.00), vec3(0.20, 0.15, 0.20)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.42, 0.11, 0.55), vec3(0.1, 0.8, 0.95));

/// The colours colonies take: green, yellow, pink, blue or purple.
fn colour(c: &CreatureInstance) -> Vec3 {
    match pick(c, 20.0, 5) {
        0 => vec3(0.12, 0.52, 0.18),
        1 => vec3(0.78, 0.62, 0.08),
        2 => vec3(0.85, 0.25, 0.45),
        3 => vec3(0.12, 0.32, 0.78),
        _ => vec3(0.42, 0.11, 0.55),
    }
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut look = LOOK;
    let pigment = colour(c);
    look.pigment = pigment.extend(0.0);
    look.accent = pigment.lerp(vec3(0.9, 0.95, 1.0), 0.55);
    super::render(ray, c, surface, look)
}
