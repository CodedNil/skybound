#define_import_path skybound::clouds
#import skybound::utils::{View, intersect_sphere}

const BASE_SCALE = 0.003;
const BASE_TIME = 0.01;

// Base Parameters
const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.2) * 0.1 * BASE_TIME; // Main wind for base shape

// Weather Parameters
const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: vec2<f32> = vec2<f32>(1.0, 0.0) * 0.02 * BASE_TIME; // Weather wind for weather shape

// Detail Parameters
const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(1.0, 0.0, -1.0) * 0.2 * BASE_TIME; // Details move faster

// Cloud scales
const CLOUD_BOTTOM_HEIGHT: f32 = 1000;
const CLOUD_TOP_HEIGHT: f32 = 48000;
const CLOUD_LAYER_HEIGHT: f32 = 3000;
const CLOUD_BASE_FRACTION: f32 = 0.2; // A lower value gives a flatter, more defined base.
const CLOUD_TOTAL_LAYERS: u32 = 16u;
// The vertical height of the layer
const CLOUD_LAYER_HEIGHTS = array<f32, CLOUD_TOTAL_LAYERS>(
    2500, 2400, 2400, 2500,
    2400, 2200, 2200, 2000,
    1800, 1500, 1100, 800,
    350, 300, 300, 250
);
// Position offset for the weather coverage per layer
const CLOUD_LAYER_OFFSETS = array<vec2<f32>, CLOUD_TOTAL_LAYERS>(
    vec2(0.8, -0.6), vec2(-0.4, 0.9), vec2(0.1, -0.8), vec2(0.6, 0.4),
    vec2(-0.2, 0.2), vec2(0.3, -0.9), vec2(0.9, -0.7), vec2(0.5, 0.1),
    vec2(-0.7, 0.3), vec2(0.0, -0.2), vec2(-0.6, 0.5), vec2(1.0, -0.5),
    vec2(0.7, -0.3), vec2(-0.1, 0.8), vec2(-0.9, -0.4), vec2(0.4, 0.6),
);
// Multiplier for the noises scale factor per layer
const CLOUD_LAYER_SCALES = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 0.9, 0.85, 0.85,
    0.8, 0.8, 0.75, 0.75,
    0.7, 0.7, 0.7, 0.6,
    0.5, 0.45, 0.3, 0.3
);
// Scale stretch on the x axis for the layer
const CLOUD_LAYER_STRETCH = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.2, 1.8,
    2.5, 3.0, 4.0, 4.0
);
// Density multiplier per layer
const CLOUD_LAYER_DENSITIES = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    0.6, 0.6, 0.6, 0.6
);
// How much to increase the detail noise in the layer
const CLOUD_LAYER_DETAILS = array<f32, CLOUD_TOTAL_LAYERS>(
    0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1,
    0.15, 0.15, 0.2, 0.25,
    0.3, 0.5, 0.6, 0.6
);


fn get_height_fraction(altitude: f32) -> f32 {
    return clamp((altitude - CLOUD_BOTTOM_HEIGHT) / (CLOUD_TOP_HEIGHT - CLOUD_BOTTOM_HEIGHT), 0.0, 1.0);
}

// Gets the current cloud layers index, bottom and top height
struct CloudLayer {
    index: u32,
    bottom: f32,
    height: f32,
}
fn get_cloud_layer(pos: vec3<f32>) -> CloudLayer {
    let index: u32 = u32((pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_HEIGHT);
    let bottom: f32 = CLOUD_BOTTOM_HEIGHT + f32(index) * CLOUD_LAYER_HEIGHT;

    // Branchless validity check, height becomes 0 if invalid
    let is_valid_layer: bool = (pos.z > CLOUD_BOTTOM_HEIGHT) && (index < CLOUD_TOTAL_LAYERS);
    let height: f32 = select(0.0, CLOUD_LAYER_HEIGHTS[index], is_valid_layer);
    let is_within_thickness: f32 = f32(pos.z <= bottom + height);

    return CloudLayer(index, bottom, height * is_within_thickness);
}

// Get midpoint height of the cloud layer above
fn get_cloud_layer_above(altitude: f32, above: f32) -> f32 {
    let above_height: f32 = altitude + CLOUD_LAYER_HEIGHT * above;
    let layer_index: u32 = u32(max((above_height - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_HEIGHT, 0.0));
    let is_valid = f32(layer_index < CLOUD_TOTAL_LAYERS);

    let bottom: f32 = CLOUD_BOTTOM_HEIGHT + f32(layer_index) * CLOUD_LAYER_HEIGHT;
    let midpoint: f32 = bottom + CLOUD_LAYER_HEIGHTS[layer_index] * 0.2;
    return mix(-1.0, midpoint, is_valid);
}

fn sample_clouds(pos: vec3<f32>, time: f32, base_texture: texture_3d<f32>, details_texture: texture_3d<f32>, weather_texture: texture_2d<f32>, linear_sampler: sampler) -> f32 {
    var cloud_layer = get_cloud_layer(pos);
    if cloud_layer.height == 0.0 { return 0.0; } // Early exit if we are not in a valid cloud layer
    let layer_i = cloud_layer.index;

    // --- Weather & Coverage ---
    let weather_pos_2d = (pos.xy + CLOUD_LAYER_OFFSETS[layer_i] * 10000.0) * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;
    let weather_noise = textureSampleLevel(weather_texture, linear_sampler, weather_pos_2d, 0.0);

    // Initial weather coverage from texture.
    var weather_coverage = mix(weather_noise.r, weather_noise.g, weather_noise.b) + mix(weather_noise.r, weather_noise.g, weather_noise.b) * 0.5;

    // Further refine coverage based on which vertical layer we are in.
    let layer_fraction = f32(layer_i) / f32(CLOUD_TOTAL_LAYERS);
    let cloud_layer_coverage_a = 1.0 - smoothstep(0.0, 0.5, distance(weather_noise.b, layer_fraction));
    let cloud_layer_coverage_b = 1.0 - smoothstep(0.0, weather_noise.b * 0.4, distance(1.0, layer_fraction));

    // Bias the clouds coverage to the highest levels
    let lf_sq = layer_fraction * layer_fraction;
    weather_coverage *= (cloud_layer_coverage_a + cloud_layer_coverage_b) * (0.5 + lf_sq * lf_sq * 0.5);
    if weather_coverage <= 0.0 { return 0.0; } // Early exit if coverage is too low

    // --- Height Gradient ---
    let cloud_height = cloud_layer.height * (0.6 + weather_noise.a * 3.0);
    let height_fraction = clamp((pos.z - cloud_layer.bottom) / cloud_height, 0.0, 1.0);
    let rise = smoothstep(0.0, CLOUD_BASE_FRACTION, height_fraction);
    let fall = smoothstep(1.0, CLOUD_BASE_FRACTION, height_fraction);
    let height_gradient = rise * fall;
    if height_gradient <= 0.0 { return 0.0; } // Early exit if density is too low

    // --- Base Cloud Shape ---
    let stretched_scale = vec3<f32>(CLOUD_LAYER_SCALES[layer_i] * CLOUD_LAYER_STRETCH[layer_i], CLOUD_LAYER_SCALES[layer_i], 1.0);
    let base_scaled_pos = pos * BASE_NOISE_SCALE * stretched_scale + time * WIND_DIRECTION_BASE;
    let base_noise = textureSampleLevel(base_texture, linear_sampler, base_scaled_pos, 0.0);
    var base_cloud = (base_noise.r * height_gradient) + weather_coverage - 1.0;
    if base_cloud <= 0.0 { return 0.0; } // Early exit if density is too low

	// --- High Frequency Detail ---
	let detail_time_vec = time * WIND_DIRECTION_DETAIL;
    let detail_stretched_scale = mix(CLOUD_LAYER_SCALES[layer_i] * max(CLOUD_LAYER_STRETCH[layer_i] * 0.2, 1.0), 1.0, -0.5);
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE * vec3<f32>(detail_stretched_scale, CLOUD_LAYER_SCALES[layer_i], 1.0) - detail_time_vec;

    let detail_noise = textureSampleLevel(details_texture, linear_sampler, detail_scaled_pos, 0.0).r;

    // Apply more less noise to the lower portion of the cloud, reduced at higher altitudes
    let detail_amount = mix(height_fraction, 1.0, CLOUD_LAYER_DETAILS[layer_i]);
    let hfbm = mix(detail_noise, 1.0 - detail_noise, clamp(detail_amount * 4.0, 0.0, 1.0));

    let erosion = hfbm * 0.4 * detail_amount;
    let inv_erosion_range = 1.0 / (1.0 - erosion + 1e-6); // Add epsilon to avoid div by zero
    base_cloud = saturate((base_cloud - erosion) * inv_erosion_range);

    return clamp(base_cloud * CLOUD_LAYER_DENSITIES[layer_i], 0.0, 1.0);
}
