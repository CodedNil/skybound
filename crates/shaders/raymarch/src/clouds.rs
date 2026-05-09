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

// --- Cloud scales
pub const CLOUD_BOTTOM_HEIGHT: f32 = 1000.0;
pub const CLOUD_TOP_HEIGHT: f32 = 33000.0;
const CLOUD_LAYER_SPACING: f32 = 1400.0;
const CLOUD_TOTAL_LAYERS: usize = 16;

const CLOUD_LAYER_HEIGHTS: [f32; 16] = [
    2200.0, 2150.0, 2100.0, 2050.0, 2000.0, 1950.0, 1900.0, 1850.0, 1750.0, 1600.0, 1500.0, 1400.0,
    1300.0, 1200.0, 1100.0, 1050.0,
];

const CLOUD_LAYER_SCALES: [f32; 16] = [
    1.0, 0.98, 0.95, 0.92, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
];

const CLOUD_LAYER_STRETCH: [f32; 16] = [
    1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3, 1.5, 1.8, 2.5, 3.5, 6.0, 10.0, 14.0, 18.0,
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
    let disp_coord = pos_xy * WEATHER_NOISE_SCALE * 0.15 + Vec2::splat(layer_seed);
    let disp_noise: f32 = weather_texture.sample(sampler, disp_coord).x;

    let variance = (disp_noise - 0.5) * CLOUD_LAYER_SPACING * 8.0;
    let dist_to_floor = index_u as f32 * CLOUD_LAYER_SPACING;

    variance.max(-dist_to_floor)
}

pub fn sample_clouds(
    pos: Vec3,
    view: &ViewUniform,
    time: f32,
    simple: bool,
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> f32 {
    let weather_uv = pos.xy() * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;
    let weather_sample: Vec4 = weather_texture.sample(sampler, weather_uv);
    let global_coverage = (weather_sample.x * 1.3 - 0.2).saturate();
    if global_coverage <= 0.0 {
        return 0.0;
    }

    let approx_idx = ((pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING).floor() as i32;

    let mut total_cloud_val: f32 = 0.0;

    // Search a wide enough range to catch any layer that could be displaced to this altitude.
    for i in -9..=9 {
        let idx_i = approx_idx + i;
        if idx_i < 0 || idx_i >= CLOUD_TOTAL_LAYERS as i32 {
            continue;
        }
        let u_idx = idx_i as u32;
        let layer_idx = u_idx as usize;

        let disp = calculate_layer_v_offset(pos.xy(), u_idx, weather_texture, sampler);

        // Add horizontal jitter per layer based on the index to break up the vertical stack
        let layer_hash = ((u_idx as f32 * 127.1).sin() * 43_758.547).fract();
        let layer_offset = vec2(
            (u_idx as f32 * 0.13 + layer_hash).cos(),
            (u_idx as f32 * 0.17 + layer_hash).sin(),
        ) * 25000.0;

        let bottom = CLOUD_BOTTOM_HEIGHT + u_idx as f32 * CLOUD_LAYER_SPACING + disp;
        let layer_h = CLOUD_LAYER_HEIGHTS[layer_idx];

        if pos.z < bottom || pos.z > bottom + layer_h {
            continue;
        }

        let is_cirrus = u_idx >= 12;
        let alt_dominance = weather_sample.y;
        let local_coverage = if layer_idx < 8 {
            global_coverage * (1.0 - alt_dominance).smoothstep(0.2, 0.6)
        } else {
            global_coverage * alt_dominance.smoothstep(0.4, 0.8)
        };

        if local_coverage <= 0.0 {
            continue;
        }

        // --- Shaping Profile ---
        let h_coord = ((pos.z - bottom) / layer_h).saturate();
        let mut h_profile = h_coord.smoothstep(0.01, 0.45) * (1.0 - h_coord).smoothstep(0.05, 0.95);

        if is_cirrus {
            h_profile = h_profile.powf(6.0);
        } else {
            h_profile = h_profile.powf(0.5.lerp(1.3, layer_idx as f32 / 16.0));
        }

        if h_profile <= 0.0 {
            continue;
        }

        let stretch_base = CLOUD_LAYER_STRETCH[layer_idx];
        let stretch = if is_cirrus {
            stretch_base * 5.0
        } else {
            stretch_base
        };
        let base_scale = BASE_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_idx];
        let base_time_vec = time * WIND_DIRECTION_BASE;

        let mut wind = WIND_DIRECTION_BASE.xy().normalize();
        if wind.length_squared() < 1e-12 {
            wind = vec2(1.0, 0.0);
        }
        let wind_perp = vec2(-wind.y, wind.x);
        let coord_pos = pos.xy() + layer_offset; // Apply the horizontal offset
        let coord_along = coord_pos.dot(wind);
        let coord_perp = coord_pos.dot(wind_perp);
        let lean = h_coord * 150.0 * stretch_base * (1.0 - layer_idx as f32 / 16.0);

        let sample_pos = if is_cirrus {
            vec3(coord_along * stretch * 6.0, coord_perp * 0.1, pos.z * 35.0) * (base_scale * 0.1)
                + base_time_vec * 0.05
        } else {
            vec3(coord_along * stretch + lean, coord_perp, pos.z) * base_scale + base_time_vec
        };

        let base_noise = base_texture.sample(sampler, sample_pos).x;
        let mut density = base_noise;
        if is_cirrus {
            density = (density * 0.8).powf(6.0) * 0.15;
        } else {
            density = (density * 1.6.lerp(2.4, weather_sample.z) - (0.1 + weather_sample.z * 0.2))
                .saturate();
        }

        let mut cloud_val = (density * h_profile) + local_coverage - 1.0;
        if cloud_val > 0.0 {
            if !simple {
                let det_pos = (pos * DETAIL_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_idx])
                    - (time * WIND_DIRECTION_DETAIL);
                let detail_noise = details_texture.sample(sampler, det_pos).x;
                let edge_mask = (1.0 - h_profile).lerp(1.0 - cloud_val, 0.5);
                let erosion =
                    detail_noise * 0.4 * CLOUD_LAYER_DETAILS[layer_idx] * (1.2 + edge_mask);
                cloud_val = (cloud_val - erosion).saturate();
            }
            total_cloud_val = total_cloud_val.max(cloud_val);
        }
    }

    total_cloud_val
}
