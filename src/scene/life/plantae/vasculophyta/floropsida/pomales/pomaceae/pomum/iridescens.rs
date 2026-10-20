use crate::scene::life::{
    CreatureInstance,
    biology::Biology,
    capsule, ellipsoid,
    plantae::rosette,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 120.0,
    lifespan_days: 3000.0,
    adult_mass: 4.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 10.0;

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
    c.opaque_bounds = vec3(5.0, 6.0, 5.0).extend(0.0);
}

fn featherpod(p: Vec3) -> Surface {
    let mut pod = Surface::new(ellipsoid(p, vec3(0.13, 0.42, 0.13)), 3);
    pod.join(rosette(p - vec3(0.0, 0.35, 0.0), 7, 0.85, 0.07, 0.45), 1);
    for i in 0..5 {
        let tip = vec3(
            (i as f32 * 2.4).cos() * 0.25,
            -0.8,
            (i as f32 * 2.4).sin() * 0.25,
        );
        pod.join(capsule(p, vec3(0.0, -0.3, 0.0), tip, 0.012), 1);
    }
    pod
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let angle = p.z.atan2(p.x);
    let rib = (angle * 9.0).cos() * 0.12;
    let mut shape = Surface::new(ellipsoid(p, vec3(2.9 + rib, 3.8, 2.9 + rib)), 3);
    shape.join(
        capsule(p, vec3(0.0, 3.2, 0.0), vec3(0.2, 4.7, 0.0), 0.17),
        0,
    );
    shape.join(rosette(p - vec3(0.0, 3.7, 0.0), 5, 3.2, 0.8, -0.7), 1);
    for i in 0..3 {
        let phase = i as f32 * 2.1 + c.animation.z * 0.1;
        let center = vec3(phase.cos() * 4.0, 1.0 + phase.sin(), phase.sin() * 4.0);
        let pod = featherpod(p - center);
        shape.join(pod.distance, pod.material);
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.82, 0.24, 0.055),
        vec3(0.95, 0.65, 0.15),
    )
}
