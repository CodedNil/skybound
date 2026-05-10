use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

use crate::utils::Smoothstep;

const BASE_SCALE: f32 = 0.005;
const BASE_TIME: f32 = 0.01;

const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: Vec3 = vec3(1.0 * 0.1 * BASE_TIME, 0.0, 0.2 * 0.1 * BASE_TIME);

const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: Vec2 = vec2(1.0 * 0.02 * BASE_TIME, 0.0);

const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: Vec3 = vec3(1.0 * 0.2 * BASE_TIME, 0.0, -0.2 * BASE_TIME);

pub const CLOUD_BOTTOM_HEIGHT: f32 = 1000.0;
const CLOUD_LAYER_SPACING: f32 = 1400.0;
const CLOUD_TOTAL_LAYERS: usize = 16;
pub const CLOUD_TOP_HEIGHT: f32 =
    CLOUD_BOTTOM_HEIGHT + CLOUD_LAYER_SPACING * CLOUD_TOTAL_LAYERS as f32;

const CLOUD_LAYER_HEIGHTS: [f32; 16] = [
    2200.0, 2150.0, 2100.0, 2050.0, 2000.0, 1950.0, 1900.0, 1850.0, 1750.0, 1600.0, 1500.0, 1400.0,
    1300.0, 1200.0, 1100.0, 1050.0,
];

const CLOUD_LAYER_SCALES: [f32; 16] = [
    1.0, 0.98, 0.95, 0.92, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
];

const CLOUD_LAYER_DETAILS: [f32; 16] = [
    0.1, 0.1, 0.1, 0.1, 0.12, 0.12, 0.15, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95,
];

fn calculate_layer_v_offset(
    pos_xy: Vec2,
    index_u: u32,
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> f32 {
    let layer_seed = index_u as f32 * 137.415;

    let large_coord = pos_xy * WEATHER_NOISE_SCALE * 0.2 + Vec2::splat(layer_seed);
    let large_noise = weather_texture.sample(sampler, large_coord).x;

    let medium_coord = pos_xy * WEATHER_NOISE_SCALE * 3.0 + Vec2::splat(layer_seed * 0.5);
    let medium_noise = weather_texture.sample(sampler, medium_coord).x;

    let detail_coord = pos_xy * WEATHER_NOISE_SCALE * 90.0 + Vec2::splat(layer_seed * 0.5);
    let detail_noise = weather_texture.sample(sampler, detail_coord).x;

    let combined_noise = large_noise * 0.9 + medium_noise * 0.1 + detail_noise * 0.01;
    (combined_noise - 0.5) * CLOUD_LAYER_SPACING * 6.0
}

pub fn sample_clouds(
    pos: Vec3,
    view: &ViewUniform,
    simple: bool,
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> f32 {
    let weather_uv = pos.xy() * WEATHER_NOISE_SCALE + view.time() * WIND_DIRECTION_WEATHER;
    let weather_sample: Vec4 = weather_texture.sample(sampler, weather_uv);
    let global_coverage = (weather_sample.x * 1.3 - 0.2).saturate();
    if global_coverage <= 0.0 {
        return 0.0;
    }

    let approx_idx = ((pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING).floor() as i32;
    let mut total_cloud_val: f32 = 0.0;

    for i in -4..=4 {
        let idx_i = approx_idx + i;
        if idx_i < 0 || idx_i >= CLOUD_TOTAL_LAYERS as i32 {
            continue;
        }
        let u_idx = idx_i as u32;
        let layer_idx = u_idx as usize;

        let disp = calculate_layer_v_offset(pos.xy(), u_idx, weather_texture, sampler);
        let bottom = CLOUD_BOTTOM_HEIGHT + u_idx as f32 * CLOUD_LAYER_SPACING + disp;
        let dynamic_height = CLOUD_LAYER_HEIGHTS[layer_idx] * (0.3 + 0.7 * global_coverage);

        if pos.z < bottom || pos.z > bottom + dynamic_height {
            continue;
        }

        let h_coord = ((pos.z - bottom) / dynamic_height).saturate();

        let base_scale = BASE_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_idx];
        let sample_pos = vec3(pos.x, pos.y, pos.z) * base_scale + view.time() * WIND_DIRECTION_BASE;
        let base_noise = base_texture.sample(sampler, sample_pos).x;

        let billow_modifier = (base_noise * 1.5).saturate();
        let perturbed_h = (h_coord - (base_noise - 0.5) * 0.4).saturate();

        // Flat bottom, rounded/billowy top — asymmetric to match real cumulus profiles
        let mut h_profile =
            perturbed_h.smoothstep(0.0, 0.2) * (1.0 - perturbed_h).powf(2.0).smoothstep(0.0, 0.7);
        h_profile *= billow_modifier;

        let density = (base_noise * 2.0 - 0.2).saturate();
        let mut cloud_val = (density * h_profile) + global_coverage - 1.0;

        if cloud_val > 0.0 {
            if !simple {
                let det_pos = (pos * DETAIL_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_idx])
                    - (view.time() * WIND_DIRECTION_DETAIL);
                let detail_noise = details_texture.sample(sampler, det_pos).x;
                cloud_val =
                    (cloud_val - detail_noise * 0.3 * CLOUD_LAYER_DETAILS[layer_idx]).saturate();
            }
            total_cloud_val = total_cloud_val.max(cloud_val);
        }
    }
    total_cloud_val
}
