use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{count, gene, pick, rotate, speckle},
    rounded,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.7,
    maturity_days: 180.0,
    lifespan_days: 5000.0,
    adult_mass: 7.0,
    altitude: (50.0, 800.0),
};

pub const RENDER_RADIUS: f32 = 16.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Stelloradix pendula",
    "Root-Stars",
    [
        "Root-Stars float on rubbery crowns while thick fleshy roots radiate downward into nutrient-rich lower air.",
        "Vascular channels lift absorbed moisture and minerals from the roots into the body.",
        "Roots reaching the upper Veiland mists suffer progressive surface damage and are shed before exposure spreads into the crown.",
        "Surviving inner tissue in a detached root curls, forms a buoyant body and rises to begin another plant.",
        "Grounders feed on these rich root structures, linking a dangerous collecting strategy to one of the lower Gardens' most specialised grazing relationships.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(8.0, 12.0, 8.0).extend(0.0);
}

/// Pieces each root is drawn in.
const PIECES: u32 = 5;

/// A root-star.
///
/// A rubbery crown floating on the updraft, from which thick, fleshy roots radiate down into the
/// rich lower air, drawing up its moisture and minerals. Roots that reach too far into the
/// Veiland mists are burned and shed, leaving stumps, and the curled tips of some swell into the
/// buds that will rise as new plants. Each grows its own: a domed crown, a flat scalloped one or a
/// lobed star; a few long roots or many short ones, hanging straight, splaying out, or twisting
/// round one another.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let angle = p.z.atan2(p.x);
    let arms = count(c, 1.0, 5, 11);
    let crown = match pick(c, 2.0, 3) {
        0 => ellipsoid(p - vec3(0.0, 0.2, 0.0), vec3(2.3, 1.6, 2.3)),
        1 => {
            let scallop = 1.0 + (angle * arms as f32).cos() * 0.12;
            rounded(vec3(p.x / scallop, p.y, p.z / scallop), vec3(3.0, 0.9, 3.0)) * 0.7
        }
        _ => {
            let star = 0.7 + (angle * arms as f32).cos().max(0.0).powi(2) * 0.6;
            rounded(vec3(p.x / star, p.y, p.z / star), vec3(2.8, 1.1, 2.8)) * star.min(1.0) * 0.6
        }
    };
    let mut shape = Surface::new(crown, 3);
    if p.y > 2.0 {
        return shape;
    }
    let long = 4.0 + gene(c, 3.0) * 6.0;
    let thick = 0.3 + gene(c, 4.0) * 0.3;
    let splay = gene(c, 5.0) * 0.8;
    let twist = (gene(c, 6.0) - 0.5) * 2.5;
    let shed = gene(c, 7.0) * 0.5;
    // Only the roots on either side of the point's bearing can reach it.
    let sector = TAU / arms as f32;
    let nearest = (angle / sector).round();
    let spread = if splay > 0.4 { 2 } else { 1 };
    for step in -spread..=spread {
        let index = nearest + step as f32;
        let bearing = index * sector;
        let short = if speckle(vec2(index, 1.0), 1.0) < shed {
            0.25
        } else {
            1.0
        };
        let reach = long * short * (0.75 + speckle(vec2(index, 2.0), 1.0) * 0.5);
        let mut previous = vec3(bearing.cos(), 0.0, bearing.sin()) * 1.2 + vec3(0.0, -0.4, 0.0);
        for k in 1..=PIECES {
            let t = k as f32 / PIECES as f32;
            let turn = bearing + twist * t;
            let out = 1.2 + splay * reach * t * 0.6;
            let sway = (c.animation.z * 0.4 + index + t * 2.0).sin() * 0.3 * t;
            let next = vec3(turn.cos() * out + sway, -0.4 - reach * t, turn.sin() * out);
            let width = thick * (1.0 - (k - 1) as f32 / PIECES as f32 * 0.7);
            shape.join(tapered_tail(p, previous, next, width, width * 0.8), 0);
            previous = next;
        }
        if short < 1.0 {
            // The burned stump where a root was shed.
            shape.join((p - previous).length() - thick * 0.7, 2);
        } else if speckle(vec2(index, 3.0), 1.0) > 0.65 {
            // A curled tip swelling into a bud.
            let q = rotate(p - previous, bearing);
            shape.join(ellipsoid(q - vec3(0.3, 0.2, 0.0), vec3(0.45, 0.5, 0.45)), 3);
        }
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.20, 0.00), vec3(0.50, 0.40, 0.50)),
    site(BLADDER, vec3(0.00, 0.30, 0.80), vec3(0.45, 0.35, 0.45)),
    pair(FROND, vec3(1.00, 0.20, 0.00), vec3(0.35, 0.30, 0.35)),
    site(ROOT, vec3(0.00, -0.40, -0.80), vec3(0.35, 0.30, 0.35)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.48, 0.27, 0.17), vec3(0.7, 0.54, 0.3));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation};

    pub const PHYSIOLOGY: Physiology = Physiology {
        propagation: Propagation::Fragmentation,
        ..super::super::PHYSIOLOGY
    };
}
