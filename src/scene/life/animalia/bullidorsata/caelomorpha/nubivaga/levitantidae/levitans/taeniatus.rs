use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.3,
    altitude: (800.0, 7000.0),
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::animalia::bullidorsata::caelomorpha::cloth;
use crate::scene::life::animalia::bullidorsata::caelomorpha::{
    GasChamber, drift_body, ribbons, tissue_distance,
};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, smooth_min};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 25.0;

pub const TAXON: Taxon = Taxon {
    name: "Levitans taeniatus",
    english_name: "Drifters",
    description: "Drifters have a flattened, leaf-like body with a pointed leading edge and long trailing ribbons. \
        Several rounded gas chambers surround a darker green cluster of gelatinous organs beneath a translucent yellow-green membrane. \
        Fine golden veins branch across the body and ribbons, whose thin margins twist into broad loops and taper to delicate curling tips. \
        They collect algae and plant fragments along their sticky surfaces, adjusting their orientation gently in response to light and airflow.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.2, 3.5, 5.2).extend(0.0);
    for i in 0..3 {
        cloth::prepare(
            c,
            i,
            vec3((i as f32 - 1.0) * 2.1, -0.25, 2.8),
            10.0 + i as f32 * 2.0,
            1.4,
        );
    }
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut distance = ellipsoid(p - vec3(0.0, -0.35, 0.0), vec3(0.9, 0.4, 1.1));
    for i in -2..=2 {
        let x = i as f32;
        for side in [-1.0, 1.0] {
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

pub fn envelope(p: Vec3, _: &CreatureInstance) -> Surface {
    let q = p - vec3(0.0, -0.2 + p.z * p.z * 0.035, 0.0);
    Surface::new(ellipsoid(q, vec3(4.3, 1.9, 4.6)) * 0.8, 0)
}

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
