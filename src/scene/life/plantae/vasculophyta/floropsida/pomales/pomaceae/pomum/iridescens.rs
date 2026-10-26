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
    plantae::{
        physiology::{BLADDER, FROND, ROOT, STEM},
        vasculophyta::floropsida::pomales::FRUIT,
    },
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.67,
    maturity_days: 120.0,
    lifespan_days: 3000.0,
    adult_mass: 4.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 10.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Pomum iridescens",
    "Fruit",
    [
        "Fruit plants gather in loose fields of buoyant bodies covered by thick iridescent rinds.",
        "The rind collects dispersed aur while vascular channels supply a fleshy interior eaten by many animals.",
        "At ripeness, the covering opens in a gentle burst rather than a violent discharge that would scatter seeds toward lethal lower air.",
        "Released featherpods use flexible rudders to adjust drag and barbed filaments to catch fur or sheltered plant growth.",
        "Animal visitors both consume the crop and carry its descendants between currents, keeping a feeding ground from becoming an isolated lineage.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 6.0, 6.0).extend(0.0);
}

/// A fruit of one of the forms the plant bears, `p` relative to its middle: round, pear-shaped,
/// ribbed like a melon, or drawn into star-like lobes. A ripe one splits along its seams, and the
/// feathered seedpods show through.
fn fruit(p: Vec3, form: u32, size: f32, ripe: bool) -> Surface {
    let angle = p.z.atan2(p.x);
    let rind = match form {
        0 => p.length() - size,
        1 => {
            // Narrowing towards the stalk.
            let narrow = 1.0 - (p.y / size).clamp(-1.0, 1.0) * 0.3;
            let q = vec3(p.x / narrow, p.y + size * 0.25, p.z / narrow);
            ellipsoid(q, vec3(1.0, 1.25, 1.0) * size) * narrow.min(1.0)
        }
        2 => p.length() - size * (1.0 - (angle * 6.0).cos().abs() * 0.08),
        _ => p.length() - size * (0.8 + (angle * 5.0).cos().max(0.0).powi(2) * 0.25),
    };
    let mut shape = Surface::new(rind, 3);
    if ripe {
        let sector = TAU / 5.0;
        let seam = ((angle / sector).round() * sector - angle).abs() * p.xz().length();
        let split = seam - 0.08 - (p.y / size + 0.3).max(0.0) * 0.15;
        shape.distance = shape.distance.max(-split);
        // Featherpods inside, their plumes ready to catch the air.
        shape.join(p.length() - size * 0.8, 1);
        let plume = vec2(p.xz().length() - size * 0.3, p.y - size * 0.9).length() - 0.05;
        shape.join(plume, 1);
    }
    shape
}

/// A fruit plant.
///
/// A buoyant core under a crown of leaves, from which fruits hang on short stalks, their thick
/// rinds shimmering as they gather the dispersed aur; roots trail below. Plants bear as their
/// seeds made them: two fruits or six, round, pear-shaped, ribbed or lobed, some already split and
/// showing their featherpods; a crown of broad leaves or of spiky ones; few roots or many.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p - vec3(0.0, 0.4, 0.0), vec3(1.1, 1.45, 1.1)), 0);
    shape.join(
        ellipsoid(p - vec3(0.0, 2.5, 0.0), vec3(0.95, 0.85, 0.95)),
        3,
    );
    // The crown.
    let leaves = count(c, 1.0, 4, 9);
    let spiky = pick(c, 2.0, 2) == 1;
    for i in 0..leaves {
        let angle = i as f32 * TAU / leaves as f32 + gene(c, 3.0);
        let q = rotate(p - vec3(0.0, 3.0, 0.0), angle);
        let blade = if spiky {
            leaf(q, 2.2 + gene(c, 4.0) * 1.2, 0.25, 1.6)
        } else {
            leaf(q, 1.8 + gene(c, 4.0) * 1.4, 0.9, 0.5)
        };
        shape.join(blade, 1);
    }
    // The fruits: always the first pair either side of the core, then any more around it.
    let fruits = count(c, 5.0, 2, 6);
    let form = pick(c, 6.0, 4);
    let ripeness = gene(c, 7.0);
    for i in 0..fruits {
        let f = i as f32;
        let (centre, size) = if i < 2 {
            (
                vec3(2.05 * (1.0 - 2.0 * f), -0.75, 0.0),
                0.85 + gene(c, 8.0) * 0.35,
            )
        } else {
            let angle = TAU / 4.0 + (f - 2.0) * TAU / 4.0 + gene(c, 9.0);
            let hang = -0.3 - speckle(vec2(f, 1.0), 1.0) * 1.8;
            (
                vec3(angle.cos() * 2.0, hang, angle.sin() * 2.0),
                0.6 + speckle(vec2(f, 2.0), 1.0) * 0.5,
            )
        };
        let stalk = vec3(centre.x * 0.35, 0.6, centre.z * 0.35);
        shape.join(
            capsule(p, stalk, centre + vec3(0.0, size * 0.8, 0.0), 0.12),
            0,
        );
        let ripe = speckle(vec2(f, 3.0), 1.0) < ripeness * 0.6;
        let one = fruit(p - centre, form, size, ripe);
        if one.distance < shape.distance {
            shape = one;
        }
    }
    for i in 0..count(c, 10.0, 3, 7) {
        let f = i as f32;
        let bearing = f * 2.39996;
        let sway = (c.animation.z * 0.5 + f).sin() * 0.2;
        let end = vec3(
            bearing.cos() * 0.7 + sway,
            -3.5 - gene(c, 11.0 + f) * 1.8,
            bearing.sin() * 0.7,
        );
        shape.join(tapered_tail(p, vec3(0.0, -0.8, 0.0), end, 0.14, 0.03), 0);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.40, 0.00), vec3(0.70, 0.90, 0.70)),
    site(BLADDER, vec3(0.00, 2.50, 0.00), vec3(0.60, 0.55, 0.60)),
    pair(FROND, vec3(0.50, -0.40, 0.00), vec3(0.30, 0.30, 0.30)),
    site(ROOT, vec3(0.00, -0.70, 0.00), vec3(0.40, 0.25, 0.40)),
    pair(FRUIT, vec3(2.05, -0.75, 0.00), vec3(0.55, 0.55, 0.55)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.82, 0.24, 0.055), vec3(0.95, 0.65, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
