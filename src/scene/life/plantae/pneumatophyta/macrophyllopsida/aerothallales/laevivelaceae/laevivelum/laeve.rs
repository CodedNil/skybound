#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use crate::scene::{
    life::{
        CreatureInstance, Look,
        biology::Biology,
        capsule, ellipsoid,
        plantae::{count, gene, pick, rotate, speckle},
        surface::{Ray, Surface, TraceResult},
        tapered_tail,
    },
    math::Power,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 1.1,
    maturity_days: 730.0,
    lifespan_days: 60000.0,
    adult_mass: 18.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 17.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Laevivelum laeve",
    "Smooth Sails",
    [
        "Smooth Sails spread broad, thin aur-collecting membranes almost horizontally through rising air.",
        "Gentle undulation changes drag and sheds uneven loads of rain before they distort the whole surface.",
        "Small marginal glands produce a soft glow where the surrounding light is dim.",
        "Overlapping sails cool and shelter the air below, creating patches useful to moisture-sensitive organisms.",
        "Windborne spores establish new membranes, while browsing Old Wise Ones keep mature sails connected to the wider food web.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(11.0, 4.0, 11.0).extend(0.0);
}

/// One sail, `p` in its own frame centred on its hub: a thin membrane spread level in the rising
/// air and rolling in slow waves that shed the rain, glowing glands along its edge. Its outline is
/// round, oval, drawn into points or into a long ribbon, as the plant grew it.
fn sail(p: Vec3, c: &CreatureInstance, reach: f32, salt: f32) -> Surface {
    let along = p.xz().length();
    let angle = p.z.atan2(p.x);
    let edge = reach
        * match pick(c, 1.0, 4) {
            0 => 1.0,
            1 => 1.0 / (1.0 + 0.6 * angle.sin().power(2)),
            2 => {
                let points = count(c, 2.0, 5, 8) as f32;
                0.72 + 0.28 * (angle * points + salt).cos().abs().powf(0.6)
            }
            _ => 1.0 / (1.0 + 2.2 * angle.sin().power(2)),
        };
    let waves = 2.0 + gene(c, 3.0) * 4.0;
    let swell = (0.1 + gene(c, 4.0) * 0.5) * (along / reach);
    let time = c.animation.z * 0.4;
    let height = (p.x / reach * waves + time + salt).sin()
        * (p.z / reach * waves * 0.7 - time).cos()
        * swell
        - (along / reach).power(2) * gene(c, 5.0) * 0.8;
    let membrane = ((p.y - height).abs() * 0.8 - 0.035).max(along - edge);
    let mut shape = Surface::new(membrane, 1);
    // Glands along the margin, glowing where the light is dim.
    let glands = 10.0 + gene(c, 6.0) * 30.0;
    let sector = TAU / glands;
    let at = (angle / sector).round() * sector;
    let rim = edge * 0.97;
    let bead = vec3(at.cos() * rim, 0.0, at.sin() * rim);
    shape.join((p - bead - vec3(0.0, height, 0.0)).length() - 0.09, 2);
    // Ribs radiating from the hub, stiffening the membrane in some.
    let ribs = count(c, 7.0, 0, 12);
    if ribs > 2 {
        let sector = TAU / ribs as f32;
        let at = (angle / sector).round() * sector;
        let across = (p.xz() - vec2(at.cos(), at.sin()) * along).length();
        let rib = vec2(across, p.y - height).length() - 0.07 * (1.0 - along / edge).max(0.2);
        shape.join(rib.max(along - edge * 0.9), 3);
    }
    shape
}

/// A smooth sail.
///
/// Broad, thin membranes spread almost level through the rising air from small hubs, with a
/// knot of gas floats under each and fine threads trailing below. One plant spreads a single
/// great sail, another a stack of overlapping ones that shade and cool the air between them, and
/// each sail rolls in waves, bears glands and ribs, and is shaped as its own.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(1e6, 1);
    let sails = count(c, 10.0, 1, 4);
    let reach = (9.5 - sails as f32 * 1.2) * (0.8 + gene(c, 11.0) * 0.25);
    for i in 0..sails {
        let f = i as f32;
        let hub = if i == 0 {
            Vec3::ZERO
        } else {
            let bearing = f * 2.3 + gene(c, 12.0) * TAU;
            vec3(bearing.cos(), 0.0, bearing.sin()) * reach * 0.6 + vec3(0.0, f * 0.7 - 1.2, 0.0)
        };
        let local = rotate(p - hub, f * 1.7 + gene(c, 13.0));
        let one = sail(local, c, reach * (1.0 - f * 0.12), f * 2.1);
        if one.distance < shape.distance {
            shape = one;
        }
        shape.join(ellipsoid(p - hub, vec3(0.55, 0.3, 0.55)), 0);
        if i > 0 {
            shape.join(capsule(p, Vec3::ZERO, hub, 0.1), 0);
        }
    }
    // Floats clustered under the first hub.
    for i in 0..count(c, 14.0, 1, 5) {
        let f = i as f32;
        let at = if i == 0 {
            vec3(0.0, -0.45, 0.0)
        } else {
            let bearing = f * TAU / 4.0;
            vec3(
                bearing.cos() * 0.5,
                -0.5 - speckle(vec2(f, 3.0), 1.0) * 0.3,
                bearing.sin() * 0.5,
            )
        };
        shape.join(ellipsoid(p - at, vec3(0.32, 0.3, 0.32)), 3);
    }
    // Threads trailing from the hub.
    for i in 0..count(c, 15.0, 0, 6) {
        let f = i as f32;
        let bearing = f * 1.1 + gene(c, 16.0) * TAU;
        let sway = (c.animation.z * 0.6 + f).sin() * 0.3;
        let end = vec3(
            bearing.cos() * 0.8 + sway,
            -2.0 - gene(c, 17.0 + f),
            bearing.sin() * 0.8,
        );
        shape.join(tapered_tail(p, vec3(0.0, -0.3, 0.0), end, 0.05, 0.015), 0);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.00, 0.00), vec3(0.20, 0.10, 0.20)),
    site(BLADDER, vec3(0.00, -0.45, 0.00), vec3(0.15, 0.12, 0.15)),
    pair(FROND, vec3(0.30, 0.00, 0.00), vec3(0.06, 0.05, 0.06)),
    site(ROOT, vec3(0.00, -0.08, 0.25), vec3(0.08, 0.06, 0.08)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.17, 0.5, 0.36), vec3(0.29, 0.95, 0.7));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
