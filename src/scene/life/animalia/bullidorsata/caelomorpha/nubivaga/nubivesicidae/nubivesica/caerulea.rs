use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::animalia::bullidorsata::caelomorpha::cloth;
use crate::scene::life::animalia::bullidorsata::caelomorpha::{
    GasChamber, drift_body, ribbons, tissue_distance,
};
use crate::scene::life::biology::Biology;
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid, smooth_min};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 120.0,
    lifespan_days: 9000.0,
    adult_mass: 3.0,
    altitude: (2500.0, 9000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 15.0;

taxon!(
    Species,
    "Nubivesica caerulea",
    "Cloud Balloons",
    [
        "Cloud Balloons float on tightly packed gas chambers enclosed by pearly jade skin and delicate yellow-green veins.",
        "A darker cluster of gelatinous organs sits above short membranes that curl beneath the body.",
        "These moist folds intercept sparse algae and plant fragments without exposing the extensive ribbons of their lower-altitude relatives.",
        "They drift mainly through cloud-bearing upper Gardens and Cloud Peaks, avoiding the driest High Drifts.",
        "As the aur cycle dims they settle into denser air, avoiding the freezing equatorial Shard Peaks where ice can lock their moist folds.",
    ],
);

const CHAMBERS: [GasChamber; 3] = [
    GasChamber {
        center: vec3(-1.7, 1.5, -0.6),
        radii: vec3(1.65, 2.6, 1.9),
    },
    GasChamber {
        center: vec3(1.7, 1.3, -0.6),
        radii: vec3(1.6, 2.3, 1.8),
    },
    GasChamber {
        center: vec3(0.0, 1.2, 2.2),
        radii: vec3(1.85, 2.1, 1.15),
    },
];

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.7, 5.5, 4.7).extend(0.0);
    for i in 0..2 {
        cloth::prepare(c, i, vec3((i as f32 - 0.5) * 2.0, -2.0, 0.6), 5.0, 1.15);
    }
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut distance = ellipsoid(p - vec3(0.0, -1.3, 0.0), vec3(0.8, 0.7, 0.7));
    for i in -2..=2 {
        let x = i as f32;
        let center = vec3(
            x * 0.55,
            -1.5 + (x * 1.9).sin() * 0.6,
            (x * 2.3).cos() * 0.65,
        );
        distance = smooth_min(distance, ellipsoid(p - center, vec3(0.7, 0.85, 0.7)), 0.4);
    }
    for i in 0..11 {
        let angle = i as f32 * 2.39996;
        let radius = (i as f32 / 11.0).sqrt() * 1.2;
        let center = vec3(
            angle.cos() * radius,
            -1.5 + (angle * 1.3).sin() * 0.55,
            angle.sin() * radius * 0.7,
        );
        distance = smooth_min(distance, ellipsoid(p - center, vec3(0.5, 0.6, 0.5)), 0.25);
    }
    Surface::new(distance + tissue_distance(p, 3.0), 0)
}

pub fn envelope(p: Vec3, _: &CreatureInstance) -> Surface {
    let q = p - vec3(0.0, 0.9, 0.3);
    let fold = (p.x * 1.8 + p.z * 0.7).sin() * (p.y * 0.8).sin() * 0.035;
    Surface::new(ellipsoid(q, vec3(3.65, 3.8, 3.35)) + fold, 0)
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = drift_body(
        ray,
        c,
        envelope,
        surface,
        vec3(0.42, 0.67, 0.24),
        3.5,
        &CHAMBERS,
    );
    ribbons(&mut result, ray, c, 2, vec3(0.38, 0.62, 0.16));
    result
}
