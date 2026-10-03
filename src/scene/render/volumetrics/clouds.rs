use crate::scene::{ViewUniform, render::utils::Textures};
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, vec2, vec3};
use spirv_std::num_traits::Float;

const BASE_SCALE: f32 = 0.005;
const BASE_TIME: f32 = 0.01;

const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: Vec3 = vec3(1.0 * 0.1 * BASE_TIME, 0.0, 0.2 * 0.1 * BASE_TIME);

const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: Vec2 = vec2(1.0 * 0.02 * BASE_TIME, 0.0);

const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: Vec3 = vec3(1.0 * 0.2 * BASE_TIME, 0.0, -0.2 * BASE_TIME);

const NOMINAL_BOTTOM: f32 = 1000.0;
pub const CLOUD_BOTTOM_HEIGHT: f32 = NOMINAL_BOTTOM - 4200.0;
const CLOUD_LAYER_SPACING: f32 = 1400.0;
const CLOUD_TOTAL_LAYERS: usize = 16;
pub const CLOUD_TOP_HEIGHT: f32 = NOMINAL_BOTTOM + CLOUD_LAYER_SPACING * 15.0 + 4200.0 + 2200.0;

fn calculate_layer_v_offset(pos_xy: Vec2, index_u: u32, textures: &Textures) -> f32 {
    let layer_seed = index_u as f32 * 137.415;

    let large_coord = pos_xy * WEATHER_NOISE_SCALE * 0.2 + Vec2::splat(layer_seed);
    let large_noise = textures.weather(large_coord).x;

    (large_noise - 0.5) * CLOUD_LAYER_SPACING * 6.0
}

pub fn sample_clouds(pos: Vec3, view: &ViewUniform, simple: bool, textures: &Textures) -> f32 {
    if pos.z < CLOUD_BOTTOM_HEIGHT || pos.z > CLOUD_TOP_HEIGHT {
        return 0.0;
    }
    let weather_uv = pos.xy() * WEATHER_NOISE_SCALE + view.time() * WIND_DIRECTION_WEATHER;
    let weather_sample: Vec4 = textures.weather(weather_uv);
    let global_coverage = (weather_sample.x * 1.3 - 0.2).saturate();
    if global_coverage <= 0.0 {
        return 0.0;
    }

    let first = ((pos.z - NOMINAL_BOTTOM - 4200.0 - 2200.0) / CLOUD_LAYER_SPACING).ceil() as i32;
    let last = ((pos.z - NOMINAL_BOTTOM + 4200.0) / CLOUD_LAYER_SPACING).floor() as i32;
    let mut total_cloud_val: f32 = 0.0;

    for idx_i in first.max(0)..=last.min(CLOUD_TOTAL_LAYERS as i32 - 1) {
        let u_idx = idx_i as u32;
        let fraction = u_idx as f32 / (CLOUD_TOTAL_LAYERS - 1) as f32;
        let height = 2200.0.lerp(1050.0, fraction);
        let scale = 1.0.lerp(0.45, fraction);

        let nominal_bottom = NOMINAL_BOTTOM + u_idx as f32 * CLOUD_LAYER_SPACING;
        if pos.z < nominal_bottom - 4200.0 || pos.z > nominal_bottom + 4200.0 + height {
            continue;
        }
        let disp = calculate_layer_v_offset(pos.xy(), u_idx, textures);
        let bottom = nominal_bottom + disp;
        let dynamic_height = height * (0.3 + 0.7 * global_coverage);

        if pos.z < bottom || pos.z > bottom + dynamic_height {
            continue;
        }

        let h_coord = ((pos.z - bottom) / dynamic_height).saturate();

        let base_scale = BASE_NOISE_SCALE * scale;
        let sample_pos = vec3(pos.x, pos.y, pos.z) * base_scale + view.time() * WIND_DIRECTION_BASE;
        let base_noise = textures.base(sample_pos).x;

        let billow_modifier = (base_noise * 1.5).saturate();
        let perturbed_h = (h_coord - (base_noise - 0.5) * 0.4).saturate();

        let mut h_profile =
            perturbed_h.smoothstep(0.0, 0.2) * (1.0 - perturbed_h).powi(2).smoothstep(0.0, 0.7);
        h_profile *= billow_modifier;

        let density = (base_noise * 2.0 - 0.2).saturate();
        let mut cloud_val = (density * h_profile) + global_coverage - 1.0;

        if cloud_val > 0.0 {
            if !simple {
                let det_pos =
                    (pos * DETAIL_NOISE_SCALE * scale) - (view.time() * WIND_DIRECTION_DETAIL);
                let detail_noise = textures.details(det_pos).x;
                cloud_val =
                    (cloud_val - detail_noise * 0.3 * 0.1.lerp(0.95, fraction.powi(3))).saturate();
            }
            total_cloud_val = total_cloud_val.max(cloud_val);
        }
    }
    total_cloud_val
}
