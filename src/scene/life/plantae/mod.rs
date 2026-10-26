use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    surface::{Material, RINGED, Ray, Surface, TraceResult, trace_material},
};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
pub mod chlorophyta;
#[cfg(not(target_arch = "spirv"))]
pub mod physiology;
pub mod pneumatophyta;
pub mod vasculophyta;

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Kingdom,
    "Plantae",
    "Plants",
    [
        "Plants build organic tissue using dispersed aur, scattered aur light, wind deformation or storm electricity.",
        "Hygroscopic surfaces recover airborne moisture, while selective tissues retain minerals from dust and rain.",
        "Some remain aloft through low mass, gas chambers or drag surfaces, and others attach to living bodies that carry their weight.",
        "Larger vascular forms connect distant tissues through flexible channels, allowing flowers, woody frameworks and hanging roots to share resources.",
        "Spores, fragments and specialised seeds disperse through currents or on animal hosts, with no soil or fixed ground stage in their life cycles.",
    ],
);

pub fn rotate(p: Vec3, angle: f32) -> Vec3 {
    let (sine, cosine) = angle.sin_cos();
    vec3(cosine * p.x - sine * p.z, p.y, sine * p.x + cosine * p.z)
}

pub fn leaf(p: Vec3, length: f32, width: f32, curl: f32) -> f32 {
    let t = (p.z / length).clamp(0.0, 1.0);
    let profile = (t * PI).sin().max(0.0).sqrt();
    let height = curl * t * t + (p.x / width * 5.0 + t * 9.0).sin() * width * 0.04 * profile;
    let lateral = p.x.abs() - width * profile;
    let axial = (-p.z).max(p.z - length);
    let thickness = (p.y - height).abs() - length * 0.003;
    lateral.max(axial).max(thickness) * 0.55
}

pub fn frond(p: Vec3, length: f32, width: f32, curl: f32, lobes: f32) -> f32 {
    let t = (p.z / length).clamp(0.0, 1.0);
    leaf(
        p,
        length,
        width * (0.77 + 0.23 * (t * lobes * TAU).cos()),
        curl,
    )
}

pub fn rosette(p: Vec3, petals: usize, length: f32, width: f32, curl: f32) -> f32 {
    let mut distance = 1e6f32;
    for i in 0..petals {
        distance = distance.min(leaf(
            rotate(p, i as f32 * TAU / petals as f32),
            length,
            width,
            curl,
        ));
    }
    distance
}

pub fn sway(mut p: Vec3, c: &CreatureInstance) -> Vec3 {
    p.y += c.life.z * p.z.abs() * 0.15;
    p.x -= (p.y * 0.4 + c.animation.z * 0.7).sin() * p.y.abs() * 0.045;
    p.z -= (p.y * 0.3 + c.animation.z * 0.5).cos() * p.y.abs() * 0.035;
    p
}

/// Bark, leaf, glowing tissue, gloss and water: a plant's materials.
pub fn plant_material(p: Vec3, id: u32, pigment: Vec3, glow: Vec3) -> Material {
    let grain = (p.y * 31.0 + (p.x * 4.0).sin()).sin() * 0.04;
    let veins = (p.x * 8.0 + (p.z * 3.0).sin())
        .sin()
        .abs()
        .smoothstep(0.9, 0.98);
    let mut material = Material::skin(pigment * (0.8 + grain + veins * 0.2), 0.48);
    match id {
        0 => {
            material.color = vec3(0.18, 0.085, 0.035) * (1.0 + grain);
            material.roughness = 0.82;
            material.texture = RINGED;
            material.relief = 0.5;
            material.grain = 0.8;
        }
        1 => {
            material.transmission = pigment.sqrt() * 0.28;
            material.ior = 1.38;
            material.translucency = 0.45;
        }
        2 => {
            material.color = glow;
            material.emission = 1.8;
            material.roughness = 0.3;
        }
        3 => {
            material.roughness = 0.22;
            material.color = pigment.lerp(glow, (p.y * 0.2 + 0.5).clamp(0.0, 1.0));
        }
        4 => {
            material.color = vec3(0.2, 0.5, 0.62);
            material.transmission = vec3(0.72, 0.88, 0.94);
            material.ior = 1.333;
            material.roughness = 0.07;
        }
        _ => {}
    }
    material
}

pub fn render<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    look: Look,
) -> TraceResult {
    let pigment = look.pigment.truncate();
    let glow = look.accent;
    let mut result = TraceResult::MISS;
    let mut offset = 0.0;
    for _ in 0..8 {
        let mut hit = trace_material(
            Ray {
                origin: ray.origin + ray.direction * offset,
                max_distance: ray.max_distance - offset,
                ..ray
            },
            c,
            &surface,
            |p, id| plant_material(p, id, pigment, glow),
        );
        if !hit.is_hit() {
            break;
        }
        hit.color_depth.w += offset;
        if hit.transmittance.max_element() < 0.01 {
            result.opaque = hit;
            break;
        }
        offset =
            hit.distance() + c.position.w * 0.008 / hit.normal.dot(ray.direction).abs().max(0.15);
        result.add(hit);
        if offset >= ray.max_distance {
            break;
        }
    }
    result
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{PLANT, Physiology};

    pub const PHYSIOLOGY: Physiology = PLANT;
}
