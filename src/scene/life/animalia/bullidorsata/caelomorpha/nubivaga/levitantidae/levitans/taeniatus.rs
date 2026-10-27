#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::animalia::bullidorsata::caelomorpha::cloth;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::bullidorsata::caelomorpha::{GasChamber, drift_body, ribbons, tissue_distance},
    biology::Biology,
    ellipsoid,
    genome::Gene,
    rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.13,
    maturity_days: 90.0,
    lifespan_days: 5400.0,
    adult_mass: 0.3,
    altitude: (800.0, 7000.0),
};

pub const RENDER_RADIUS: f32 = 25.0;

/// How long its feeding ribbons trail, how finely its lens is drawn to a point, and how broad the
/// lip they hang from.
///
/// Long ribbons reach further into the neighbouring streams, and a fine point turns it more readily
/// into useful air.
pub const RIBBONS: Gene = Gene(38);
pub const KEEL: Gene = Gene(39);
pub const LIP: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Levitans taeniatus",
    "Drifters",
    [
        "Drifters carry a translucent yellow-green body ahead of long ribbons traced with golden veins.",
        "Several rounded gas chambers surround a darker cluster of organs, keeping the pointed front poised in the updraft.",
        "The ribbons curl through adjacent currents and collect algae and plant fragments on their sticky margins.",
        "Gentle changes in orientation regulate collection, while folded ribbons reduce strain when the animal crosses a faster stream.",
        "Their broad altitude range lets them follow suspended food upward and settle toward richer air when it thins.",
    ],
);

const CHAMBERS: [GasChamber; 3] = [
    GasChamber {
        center: vec3(-1.65, 0.55, 0.0),
        radii: vec3(0.8, 1.05, 1.65),
    },
    GasChamber {
        center: vec3(0.0, 0.65, -0.35),
        radii: vec3(0.95, 1.2, 1.8),
    },
    GasChamber {
        center: vec3(1.7, 0.45, 0.3),
        radii: vec3(0.8, 1.0, 1.5),
    },
];

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.2, 3.5, 5.2).extend(0.0);
    for i in 0..3 {
        cloth::prepare(
            c,
            i,
            vec3((i as f32 - 1.0) * 2.1, -0.45, 4.5),
            (10.0 + i as f32 * 2.0) * (0.8 + c.gene(RIBBONS) * 0.4),
            1.4,
        );
    }
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut distance = ellipsoid(p - vec3(0.0, -0.35, 0.0), vec3(0.9, 0.4, 1.1));
    for i in -2..=2 {
        let x = i as f32;
        for half in 0..2 {
            let side = if half == 0 { -1.0 } else { 1.0 };
            let center = vec3(
                x * 0.7,
                -0.5 + (x * 2.0).sin() * 0.3,
                side * (0.8 - x.abs() * 0.15),
            );
            distance = smooth_min(distance, ellipsoid(p - center, vec3(0.65, 0.8, 0.65)), 0.35);
        }
    }
    Surface::new(distance + tissue_distance(p, 3.0), 0)
}

/// A flattened lens drawn out to a pointed leading edge.
///
/// The current presses unevenly across it and turns it to face useful air; behind, it thins to a
/// lip along which the feeding ribbons are rooted, trailing clear of the body into the
/// neighbouring streams.
pub fn envelope(p: Vec3, c: &CreatureInstance) -> Surface {
    let front = (-p.z / 4.6).clamp(0.0, 1.0);
    let radii = vec3(
        4.3 * (1.0 - (0.6 + c.gene(KEEL) * 0.25) * front * front),
        1.9 * (1.0 - 0.55 * front),
        4.6,
    );
    let lens = rounded(p - vec3(0.0, -0.2 + p.z * p.z * 0.03, 0.0), radii) * 0.75;
    let lip = rounded(
        p - vec3(0.0, -0.45, 3.9),
        vec3(3.0 + c.gene(LIP) * 1.0, 0.35, 0.9),
    );
    Surface::new(smooth_min(lens, lip, 0.6), 0)
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(FLOAT, vec3(0.00, -0.36, 0.14), vec3(0.29, 0.29, 0.29)),
    pair(FLOAT, vec3(1.02, -0.43, 0.28), vec3(0.19, 0.19, 0.19)),
    site(CROP, vec3(0.00, -0.64, -0.42), vec3(0.23, 0.23, 0.23)),
    pair(BREATHER, vec3(1.21, -0.71, -0.28), vec3(0.29, 0.15, 0.29)),
    site(BROOD, vec3(0.00, -0.78, 0.42), vec3(0.20, 0.12, 0.20)),
]);

/// Tint and scale, shared by the preview and the world renderer.
pub const LOOK: Look = Look::drift(vec3(0.38, 0.66, 0.19), 3.8);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = drift_body(
        ray,
        c,
        envelope,
        surface,
        vec3(0.38, 0.66, 0.19),
        3.8,
        &CHAMBERS,
    );
    ribbons(&mut result, ray, c, 3, vec3(0.42, 0.66, 0.16));
    result
}
