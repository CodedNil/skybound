use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, ellipsoid,
    plantae::{count, frond, gene, pick, rosette, speckle},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, pair, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.2,
    maturity_days: 5.0,
    lifespan_days: 1000.0,
    adult_mass: 0.05,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 9.0;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Aeromuscus velutinus",
    "Air Moss",
    [
        "Air Moss carpets bark and rough animal shells with dense green leaves that hold dew between their tips.",
        "The mat uses scattered aur light and dissolved nutrients retained from dust and decaying debris.",
        "Its low profile keeps most growth within the sheltered air close to a moving host.",
        "Tiny animals find moisture and refuge among the leaves, while slow decomposition returns their waste to the plant.",
        "Detached pieces can start another patch, making close passages between living communities important opportunities for dispersal.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 3.0, 5.0).extend(0.0);
}

/// How far out from the patch's middle a point lies, one at its edge. A patch spreads as the
/// surface it grew on let it: a round cushion, a long strip along a crease of bark or shell, or a
/// lobed mat that crept out along several ridges at once.
fn reach(c: &CreatureInstance, at: Vec2) -> f32 {
    let form = pick(c, 1.0, 3);
    let stretch = if form == 1 {
        vec2(1.35, 0.55)
    } else {
        Vec2::ONE
    } * (0.8 + gene(c, 4.0) * 0.3)
        * 3.0;
    let lobes = if form == 2 {
        let many = count(c, 6.0, 3, 6) as f32;
        0.75 + 0.25 * (at.y.atan2(at.x) * many + gene(c, 7.0) * TAU).cos()
    } else {
        1.0
    };
    (at / stretch).length() / lobes
}

/// A mat of air moss.
///
/// A low mound of tangled stems, thickest in the middle where the patch first took hold, covered
/// in tufts that hold dew between their tips. Patches grow as their spores' genes and their
/// surfaces let them: some tufts open into starry rosettes, some rise as feathery fronds, some
/// stay tight round knobs; some patches send up stalked spore capsules, and some hold beads of
/// dew that tiny animals drink.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let height = 0.35 + gene(c, 5.0) * 0.7;
    let top = |at: Vec2| {
        let out = reach(c, at);
        height * (1.0 - out * out).max(0.0) - 0.4
    };
    let out = reach(c, p.xz());
    let under = -0.42 - 0.33 * (1.0 - out * out).max(0.0);
    let mat = ((out - 1.0) * 2.0).max(p.y - top(p.xz())).max(under - p.y) * 0.6;
    let mut shape = Surface::new(mat, 1);
    if p.y > top(p.xz()) + 1.2 && p.y > 0.9 {
        return shape;
    }
    // The tufts, one to each cell of a grid scattered over the mat.
    let tuft = pick(c, 2.0, 3);
    let spacing = 0.42;
    let cell = (p.xz() / spacing).round();
    let jitter = vec2(speckle(cell, 1.0), speckle(cell, 2.0)) - 0.5;
    let at = (cell + jitter * 0.6) * spacing;
    if reach(c, at) < 1.0 {
        let root = vec3(at.x, top(at), at.y);
        let tall = (0.2 + speckle(cell, 3.0) * 0.3) * (0.6 + height);
        let tip = root + vec3(jitter.x * 0.2, tall, jitter.y * 0.2);
        let q = p - root;
        let growth = match tuft {
            0 => rosette(p - tip, 5, 0.26, 0.07, 0.35).min(capsule(p, root, tip, 0.035)),
            1 => frond(vec3(q.x, q.z, q.y), tall * 1.7, 0.09, 0.0, 3.0),
            _ => ellipsoid(
                p - root - vec3(0.0, tall * 0.4, 0.0),
                vec3(0.17, tall * 0.55, 0.17),
            ),
        };
        shape.join(growth, 1);
        if speckle(cell, 4.0) > 0.82 - gene(c, 8.0) * 0.3 {
            shape.join((p - tip - vec3(0.0, 0.05, 0.0)).length() - 0.06, 4);
        }
    }
    // Spore capsules on fine stalks, nodding in the moving air.
    for i in 0..count(c, 3.0, 0, 6) {
        let f = i as f32;
        let angle = f * 2.39996 + gene(c, 9.0) * TAU;
        let at = vec2(angle.cos(), angle.sin()) * (0.4 + (f * 0.37).fract() * 1.5);
        let root = vec3(at.x, top(at), at.y);
        let nod = vec3(
            (c.animation.z * 0.8 + f).sin() * 0.12,
            0.9 + gene(c, 10.0 + f) * 0.6,
            0.15,
        );
        shape.join(capsule(p, root, root + nod, 0.025), 0);
        shape.join(ellipsoid(p - root - nod, vec3(0.07, 0.12, 0.07)), 3);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, -0.50, 0.00), vec3(0.32, 0.12, 0.32)),
    site(BLADDER, vec3(0.00, -0.35, -0.40), vec3(0.12, 0.08, 0.12)),
    pair(FROND, vec3(1.00, -0.55, 0.00), vec3(0.14, 0.08, 0.14)),
    site(ROOT, vec3(0.00, -0.68, 0.00), vec3(0.14, 0.05, 0.14)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.12, 0.36, 0.07), vec3(0.45, 0.65, 0.15));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}
