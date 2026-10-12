use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::biology::{Anatomy, Biology, Reproduction};
use super::{CreatureInstance, ellipsoid, smooth_min};
use crate::scene::surface::{Material, Ray, Surface, TraceResult, trace_material};

pub const DAY: f32 = 86_400.0;

pub fn grid_age(biology: Biology, panel: usize) -> f32 {
    if panel < 6 {
        biology.maturity_days * panel as f32 / 5.0
    } else {
        biology.maturity_days
            + (biology.lifespan_days - biology.maturity_days) * (panel - 5) as f32 / 3.0
    }
}

#[cfg(not(target_arch = "spirv"))]
pub fn prepare(c: &mut CreatureInstance, biology: Biology, age_days: f32, seed: u32) {
    let random = |channel: u32| {
        if seed == 0 {
            return 0.5;
        }
        let mut value = seed ^ channel.wrapping_mul(0x9e37_79b9);
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        ((value ^ (value >> 16)) >> 8) as f32 / 16_777_215.0
    };
    let age = age_days.max(0.0);
    let young = (age / biology.maturity_days).clamp(0.0, 1.0);
    let old = age.smoothstep(biology.lifespan_days * 0.72, biology.lifespan_days);
    let stage = if young >= 0.7 {
        0.0
    } else {
        match (biology.anatomy, biology.reproduction) {
            (Anatomy::Porous, _) => 1.0,
            (Anatomy::Suctorial, _) => 2.0,
            (Anatomy::Tetralate, Reproduction::Eggs) => {
                if young < 0.42 {
                    3.0
                } else {
                    4.0
                }
            }
            (Anatomy::Plant, Reproduction::Division) => 7.0,
            (Anatomy::Plant, Reproduction::Spores | Reproduction::Pollination) if young < 0.2 => {
                5.0
            }
            (Anatomy::Plant, _) => 6.0,
            _ => 0.0,
        }
    };
    c.kind.y = seed;
    c.life = vec4(age, young, old, stage);
    c.identity = vec4(random(1), random(2), random(3), random(4));
    c.animation.z += (random(5) - 0.5) * TAU;
    let growth = (0.25 + young * 0.75) * (1.0 - old * 0.08);
    let variation = (c.identity.xyz() - 0.5) * 0.35 + 1.0;
    c.proportions = (variation
        * vec3(
            1.25 - young * 0.25,
            1.15 - young * 0.15,
            0.65 + young * 0.35,
        )
        * growth
        * vec3(
            1.0 + (c.animation.z * 1.4).sin() * 0.012,
            1.0 + (c.animation.z * 1.4).sin() * 0.02,
            1.0,
        ))
    .extend(young.smoothstep(0.55, 1.0));
    c.animation.x *= 1.0 - old * 0.45;
}

pub fn scale(c: &CreatureInstance) -> Vec3 {
    if c.proportions.x > 0.0 {
        c.proportions.xyz()
    } else {
        Vec3::ONE
    }
}

pub fn body_point(mut point: Vec3, c: &CreatureInstance) -> Vec3 {
    let infant = 1.0 - c.life.y;
    let front = (-point.z / c.opaque_bounds.z.max(0.01)).smoothstep(0.25, 0.8);
    point.x *= 1.0 - infant * front * 0.15;
    point.y *= 1.0 - infant * front * 0.12;
    point
}

pub fn appearance(mut material: Material, point: Vec3, c: &CreatureInstance) -> Material {
    if c.proportions.x <= 0.0 {
        return material;
    }
    let individual = c.identity - 0.5;
    let pigment = material
        .color
        .lerp(material.color.yzx(), individual.x.abs() * 0.45);
    let pattern = (point.z * (2.0 + c.identity.y * 5.0)
        + (point.x * 3.0 + individual.z * TAU).sin() * 2.0)
        .sin();
    material.color = pigment * (1.0 + individual.w * 0.3 + pattern * individual.y * 0.16);
    material.color = material
        .color
        .lerp(material.color.sqrt(), (1.0 - c.life.y) * 0.4);
    material.color = material.color.lerp(vec3(0.42, 0.34, 0.27), c.life.z * 0.35);
    material.roughness =
        (material.roughness + individual.z * 0.12 + c.life.z * 0.18).clamp(0.04, 1.0);
    material.emission *= (0.3 + c.life.y * 0.7) * (1.0 - c.life.z * 0.65);
    material
}

pub fn juvenile_surface(point: Vec3, c: &CreatureInstance) -> Surface {
    let extent = c.opaque_bounds.xyz() * 0.85;
    let mut p = point / extent;
    let time = c.animation.z;
    let age = c.life.y;
    p.x -= (p.z * 4.0 - time * 2.0).sin() * 0.08 * p.z;
    let mut d = 1e6f32;
    let mut id = 0;
    match c.life.w as u32 {
        1 => {
            d = ellipsoid(p, vec3(0.7, 0.4 + age * 0.3, 0.75));
            let fringe = ((p.x.hypot(p.z) - 0.64).hypot(p.y + 0.2)
                - 0.08
                - (p.x.atan2(p.z) * 24.0 + time * 3.0).sin() * 0.025)
                .max(p.y);
            d = smooth_min(d, fringe, 0.12);
            for index in 0..3 {
                let chamber = ellipsoid(
                    p - vec3((index as f32 - 1.0) * 0.35, 0.25, 0.0),
                    vec3(0.25, age * 0.6 + 0.05, 0.35),
                );
                d = smooth_min(d, chamber, 0.15);
            }
        }
        2 | 3 => {
            for index in 0..7 {
                let t = index as f32 / 6.0;
                let radius = 0.20 + (t * PI).sin() * 0.13;
                d = smooth_min(
                    d,
                    ellipsoid(
                        p - vec3(0.0, 0.0, (t - 0.5) * 1.35),
                        vec3(radius, radius * 0.85, 0.19),
                    ),
                    0.06,
                );
            }
            let sucker = ellipsoid(p - vec3(0.0, 0.0, -0.67), vec3(0.25, 0.22, 0.12));
            d = smooth_min(d, sucker, 0.04);
            for pair in 0..2 {
                let q = vec3(
                    p.x.abs() - 0.28,
                    p.y + 0.02,
                    p.z - (pair as f32 - 0.5) * 0.5,
                );
                d = d.min(ellipsoid(q, vec3(0.12 + age * 0.3, 0.035, 0.19)));
            }
        }
        4 => {
            let ridges = (p.z * 60.0 + p.x.atan2(p.y) * 4.0).sin() * 0.009;
            d = ellipsoid(p, vec3(0.46, 0.4, 0.85)) + ridges;
            id = 1;
        }
        5 => {
            d = ellipsoid(p, vec3(0.3, 0.45, 0.3));
            let sail = ellipsoid(p - vec3(0.0, 0.4, 0.0), vec3(0.7, 0.06, 0.65));
            d = smooth_min(d, sail, 0.06);
            id = 2;
        }
        6 => {
            d = ellipsoid(p, vec3(0.35, 0.3, 0.35));
            let stalk = ellipsoid(p - vec3(0.0, 0.3, 0.0), vec3(0.06, 0.5, 0.06));
            d = smooth_min(d, stalk, 0.06);
            for index in 0..4 {
                let angle = index as f32 * TAU / 4.0;
                let q =
                    super::plantae::rotate(p - vec3(0.0, 0.1 + index as f32 * 0.12, 0.0), angle);
                d = d.min(super::plantae::leaf(
                    q,
                    0.35 + age * 0.75,
                    0.14 + age * 0.22,
                    0.08 + (time + angle).sin() * 0.04,
                ));
            }
            id = 3;
        }
        _ => {
            d = smooth_min(
                ellipsoid(p - vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
                ellipsoid(p + vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
                0.15,
            );
            id = 3;
        }
    }
    Surface::new(d * extent.min_element() * 0.7, id)
}

pub fn render_juvenile(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::MISS;
    result.opaque = trace_material(ray, c, juvenile_surface, |_, id| {
        let color = match id {
            1 => vec3(0.64, 0.49, 0.26),
            2 => vec3(0.40, 0.26, 0.12),
            3 => vec3(0.24, 0.59, 0.37),
            _ => vec3(0.58, 0.75, 0.65),
        };
        let mut material = Material::skin(color, if id == 1 { 0.7 } else { 0.3 });
        material.translucency = 0.35;
        material
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{KINDS, Kind};
    use spirv_std::glam::Vec4;

    #[test]
    fn lifecycle_grid_includes_birth_maturity_and_end_of_life() {
        for kind in KINDS {
            let biology = kind.biology();
            assert!(grid_age(biology, 0).abs() < f32::EPSILON);
            assert!((grid_age(biology, 5) - biology.maturity_days).abs() < 0.1);
            assert!((grid_age(biology, 8) - biology.lifespan_days).abs() < 0.1);
            for panel in 1..9 {
                assert!(grid_age(biology, panel) > grid_age(biology, panel - 1));
            }
        }
    }

    #[test]
    fn identity_is_stable_and_development_changes_body_plan() {
        for kind in [
            Kind::Nae,
            Kind::Nebulaphora,
            Kind::Nubivesica,
            Kind::Polliniflora,
        ] {
            let make = |age, seed| {
                CreatureInstance::specimen(
                    Vec4::W * kind.render_radius(),
                    Vec4::W,
                    Vec4::ZERO,
                    kind as u32,
                    age,
                    seed,
                )
            };
            let a = make(0.0, 123);
            let b = make(kind.biology().maturity_days, 123);
            assert_eq!(a.identity, b.identity);
            assert_ne!(a.identity, make(0.0, 456).identity);
            assert!(a.life.y < b.life.y);
            if kind != Kind::Nae {
                assert!(a.life.w > 0.0 && b.life.w == 0.0);
            }
        }
    }
}
