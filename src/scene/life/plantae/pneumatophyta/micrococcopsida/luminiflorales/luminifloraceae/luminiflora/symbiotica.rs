use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{count, gene, leaf, pick, rotate, speckle},
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Function, Organ, Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.19,
    maturity_days: 2.0,
    lifespan_days: 120.0,
    adult_mass: 0.03,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 7.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Luminiflora symbiotica",
    "Lumaweed",
    [
        "Lumaweed grows as rapidly expanding luminous patches on kelp, animal coverings and other living surfaces.",
        "It collects dispersed aur and releases a soft glow in dim skies, making occupied branches visible in mineral haze.",
        "Fragments readily establish new patches when neighbouring organisms brush together.",
        "Animal hosts receive some of its nutrients, but plant hosts lose resources and collecting space to the same growth.",
        "Small grazers harvest the edible tissue, limiting its spread and carrying production from a host-bound patch into the surrounding food web.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 3.0, 4.0).extend(0.0);
}

/// A patch of lumaweed.
///
/// A soft crust spreading over its host, from which glowing bulbs rise on short stalks, lighting
/// the branches it grows on in the dim skies. Patches grow their bulbs their own way: tight in a
/// cluster, in a ring round the patch's first growth, or scattered over it; as round orbs, as
/// teardrops, or as open cups whose glow lies inside; on short stalks or long; and some put out
/// curling fronds between them.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let reach = 2.0 + gene(c, 1.0) * 1.3;
    let angle = p.z.atan2(p.x);
    let lobes = 0.85 + 0.15 * (angle * count(c, 2.0, 2, 5) as f32 + gene(c, 3.0) * TAU).sin();
    let crust = ellipsoid(
        vec3(p.x, p.y + 0.3, p.z) / vec3(lobes, 1.0, lobes),
        vec3(reach, 0.32, reach),
    ) * lobes.min(1.0);
    let mut shape = Surface::new(crust, 1);
    let bulbs = count(c, 4.0, 6, 20);
    let layout = pick(c, 5.0, 3);
    let form = pick(c, 6.0, 3);
    let tall = 0.2 + gene(c, 7.0) * 1.4;
    let size = 0.16 + gene(c, 8.0) * 0.24;
    for i in 0..bulbs {
        let f = i as f32;
        let at = match layout {
            0 => (f / bulbs as f32).sqrt() * reach * 0.45,
            1 => reach * 0.6,
            _ => (f / bulbs as f32).sqrt() * reach * 0.85,
        };
        let bearing = f * if layout == 1 {
            TAU / bulbs as f32
        } else {
            2.39996
        };
        let foot = vec3(bearing.cos() * at, -0.05, bearing.sin() * at);
        let lean = vec3(foot.x, 0.0, foot.z) * 0.15;
        let height = tall * (0.5 + speckle(vec2(f, 1.0), 1.0));
        let head = foot + lean + vec3((c.animation.z * 0.8 + f).sin() * 0.05, height, 0.0);
        if (p - head).length() > 1.2 && (p - foot).length() > height + 1.0 {
            continue;
        }
        shape.join(tapered_tail(p, foot, head, 0.07, 0.045), 1);
        let bulb = size * (0.7 + speckle(vec2(f, 2.0), 1.0) * 0.6);
        let q = p - head;
        let glow = match form {
            0 => q.length() - bulb,
            1 => ellipsoid(
                q - vec3(0.0, bulb * 0.6, 0.0),
                vec3(bulb * 0.7, bulb * 1.5, bulb * 0.7),
            ),
            _ => {
                let cup = (q - vec3(0.0, bulb * 0.4, 0.0)).length() - bulb;
                let hollow = (q - vec3(0.0, bulb * 1.0, 0.0)).length() - bulb * 0.85;
                shape.join(cup.max(-hollow), 1);
                (q - vec3(0.0, bulb * 0.35, 0.0)).length() - bulb * 0.45
            }
        };
        shape.join(glow, 2);
    }
    for i in 0..count(c, 9.0, 0, 8) {
        let f = i as f32;
        let bearing = f * 2.1 + gene(c, 10.0);
        let at = vec3(bearing.cos(), 0.0, bearing.sin()) * reach * (0.3 + (f * 0.29).fract() * 0.5);
        let local = rotate(p - at, bearing);
        shape.join(leaf(vec3(local.x, local.z, local.y), 0.9, 0.18, 0.4), 1);
    }
    shape
}

/// The glow gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const GLOW: Organ = Organ::new("glow gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.30, 0.00), vec3(0.30, 0.12, 0.30)),
    site(BLADDER, vec3(0.00, -0.25, 0.50), vec3(0.15, 0.08, 0.15)),
    pair(FROND, vec3(0.90, -0.30, 0.00), vec3(0.12, 0.06, 0.12)),
    site(ROOT, vec3(0.00, -0.48, 0.00), vec3(0.15, 0.05, 0.15)),
    site(GLOW, vec3(0.00, -0.22, -0.50), vec3(0.12, 0.06, 0.12)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.09, 0.4, 0.22), vec3(0.15, 0.95, 0.8));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation, epiphyte};

    pub const PHYSIOLOGY: Physiology = Physiology {
        feeds_host: true,
        propagation: Propagation::Fragmentation,
        attaches: true,
        partner: epiphyte,
        ..super::super::PHYSIOLOGY
    };
}
