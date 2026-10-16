use spirv_std::glam::{FloatExt, Vec3};
use spirv_std::num_traits::Float;

use crate::scene::climate::clouds::storm_at;
use crate::scene::render::utils::Textures;

pub fn sample_storm(
    point: Vec3,
    time: f32,
    strength: f32,
    footprint: f32,
    textures: &Textures,
) -> f32 {
    if strength <= 0.001 {
        return 0.0;
    }
    let Some(storm) = storm_at(point, time) else {
        return 0.0;
    };
    let local = storm.coordinates(point);
    let bound = local.length_squared();
    if bound >= 1.0 {
        return 0.0;
    }
    let height = local.dot(storm.up);
    let radial = (local - storm.up * height).length();
    let anvil = height.smoothstep(0.1, 0.65);
    let width = 0.32 + anvil * 0.58 + (height * 7.0 + time * 0.015).sin() * 0.04;
    let flow = point * 0.000_08 - storm.up * time * 0.003;
    let noise = textures.billows(flow, footprint * 0.000_08);
    let body = (width + (noise.x - 0.65) * 0.45 - radial).smoothstep(-0.12, 0.12);
    let boundary = (1.0 - bound - (1.0 - noise.x) * 0.9).smoothstep(0.0, 0.2);
    body * boundary * strength * (0.2 + noise.y * 0.8) * 0.3
}
