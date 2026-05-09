#define_import_path skybound::clouds
#import skybound::utils::{View, intersect_sphere, latitude}

const BASE_SCALE = 0.005;
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
const CLOUD_BOTTOM_HEIGHT: f32 = 1000.0;
const CLOUD_TOP_HEIGHT: f32 = 33000.0;
const CLOUD_LAYER_SPACING: f32 = 2000.0; // The grid spacing for layers
const CLOUD_TOTAL_LAYERS: u32 = 16u;

// The vertical height of the layer
const CLOUD_LAYER_HEIGHTS = array<f32, CLOUD_TOTAL_LAYERS>(
    1950.0, 1900.0, 1850.0, 1800.0,
    1750.0, 1600.0, 1550.0, 1500.0,
    1400.0, 1350.0, 1200.0, 1100.0,
    900.0, 850.0, 700.0, 680.0
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
    1.0, 1.0, 1.0, 1.1,
    1.1, 1.2, 1.2, 1.3,
    1.5, 1.8, 2.5, 3.5,
    6.0, 10.0, 14.0, 18.0
);
// How much to increase the detail noise in the layer
const CLOUD_LAYER_DETAILS = array<f32, CLOUD_TOTAL_LAYERS>(
    0.1, 0.1, 0.1, 0.1,
    0.15, 0.15, 0.15, 0.15,
    0.2, 0.25, 0.3, 0.4,
    0.6, 0.8, 0.9, 0.95
);

struct CloudLayer {
    index: u32,
    bottom: f32,
    height: f32,
    is_cirrus: bool,
    displacement: f32,
};

fn calculate_layer_v_offset(pos_xy: vec2<f32>, index_u: u32, weather_texture: texture_2d<f32>, linear_sampler: sampler) -> f32 {
    let disp_coord = pos_xy * 0.000005 + vec2<f32>(f32(index_u) * 0.23);
    let disp_noise = textureSampleLevel(weather_texture, linear_sampler, disp_coord, 0.0).r;
    return (disp_noise - 0.5) * CLOUD_LAYER_SPACING * 0.85;
}

fn get_cloud_layer(pos: vec3<f32>, weather_texture: texture_2d<f32>, linear_sampler: sampler, time: f32) -> CloudLayer {
    let raw_index: f32 = (pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING;
    let index_u: u32 = u32(max(0.0, floor(raw_index)));

    if (pos.z < CLOUD_BOTTOM_HEIGHT - CLOUD_LAYER_SPACING || index_u >= CLOUD_TOTAL_LAYERS) {
        return CloudLayer(0u, 0.0, 0.0, false, 0.0);
    }

    let displacement = calculate_layer_v_offset(pos.xy, index_u, weather_texture, linear_sampler);
    let layer_bottom = CLOUD_BOTTOM_HEIGHT + f32(index_u) * CLOUD_LAYER_SPACING + displacement;
    let height = CLOUD_LAYER_HEIGHTS[index_u];

    let is_within: bool = (pos.z >= layer_bottom) && (pos.z <= layer_bottom + height);
    let is_cirrus = index_u >= 12u;

    return CloudLayer(index_u, layer_bottom, select(0.0, height, is_within), is_cirrus, displacement);
}

fn get_cloud_layer_above(altitude: f32, above_count: f32) -> f32 {
    let target_index: i32 = i32(floor((altitude - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING + above_count));
    if (target_index < 0 || target_index >= i32(CLOUD_TOTAL_LAYERS)) {
        return -1.0;
    }
    return CLOUD_BOTTOM_HEIGHT + f32(target_index) * CLOUD_LAYER_SPACING;
}

fn sample_clouds(pos: vec3<f32>, view: View, time: f32, simple: bool, base_texture: texture_3d<f32>, details_texture: texture_3d<f32>, weather_texture: texture_2d<f32>, linear_sampler: sampler) -> f32 {
    let cloud_layer = get_cloud_layer(pos, weather_texture, linear_sampler, time);
    if cloud_layer.height <= 0.0 { return 0.0; }

    let layer_i = cloud_layer.index;

    // --- Weather & Coverage ---
    let weather_pos_2d = (pos.xy + CLOUD_LAYER_OFFSETS[layer_i] * 10000.0) * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;
    let weather_noise = textureSampleLevel(weather_texture, linear_sampler, weather_pos_2d, 0.0);

    var weather_coverage = weather_noise.r * 1.2 - 0.4;
    let layer_fraction = f32(layer_i) / f32(CLOUD_TOTAL_LAYERS);
    let cloud_layer_coverage_a = 1.0 - smoothstep(0.0, 0.6, distance(weather_noise.b, layer_fraction));
    let cloud_layer_coverage_b = 1.0 - smoothstep(0.0, weather_noise.b * 0.5, distance(1.0, layer_fraction));
    weather_coverage *= (cloud_layer_coverage_a + cloud_layer_coverage_b);
    weather_coverage *= (1.0 - layer_fraction * 0.3);

    let lat_norm = saturate(abs(latitude(view)) * 0.8);
    weather_coverage *= mix(0.5, 1.0, lat_norm);

    if weather_coverage <= 0.0 { return 0.0; }

    // --- Height Gradient & Shaping ---
    let cloud_height = cloud_layer.height * (0.8 + weather_noise.a * 1.5);
    let h_coord = clamp((pos.z - cloud_layer.bottom) / cloud_height, 0.0, 1.0);

    // SMOOTHER AND STRONGER FALLOFF:
    // We use a combination of smoothstep and power functions to create a more rounded/natural density profile.
    var h_profile: f32 = 0.0;
    if (cloud_layer.is_cirrus) {
        // High clouds: Very smooth, thin wisp
        h_profile = pow(smoothstep(0.0, 0.4, h_coord) * smoothstep(1.0, 0.3, h_coord), 1.5);
    } else if (layer_i < 5u) {
        // Cumulus: Stronger base falloff to avoid flatness, very smooth top
        let base = smoothstep(0.0, 0.15, h_coord);
        let top = smoothstep(1.0, 0.4, h_coord);
        h_profile = pow(base * top, 0.7); // 0.7 power makes the middle "fuller" while keeping ends tapered
    } else {
        // Stratus/Altocumulus: Balanced smooth profile
        let base = smoothstep(0.0, 0.2, h_coord);
        let top = smoothstep(1.0, 0.2, h_coord);
        h_profile = base * top;
    }

    // Additional global shaping to ensure no hard edges at layer boundaries
    h_profile *= smoothstep(0.0, 0.1, h_coord) * smoothstep(1.0, 0.9, h_coord);

    if h_profile <= 0.0 { return 0.0; }

    // --- Base Cloud Shape ---
    let stretch = CLOUD_LAYER_STRETCH[layer_i];
    let base_scale = BASE_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_i];
    let base_time = time * WIND_DIRECTION_BASE;

    var wind = normalize(WIND_DIRECTION_BASE.xy);
    if (length(wind) < 1e-6) { wind = vec2<f32>(1.0, 0.0); }
    let wind_perp = vec2<f32>(-wind.y, wind.x);
    let coord_along = dot(pos.xy, wind);
    let coord_perp = dot(pos.xy, wind_perp);
    let base_time_vec = vec3<f32>(dot(base_time.xy, wind), dot(base_time.xy, wind_perp), base_time.z);

    if simple {
        let shadow_pos = vec3<f32>(coord_along * stretch, coord_perp, pos.z) * base_scale + base_time_vec;
        let shadow_noise = textureSampleLevel(base_texture, linear_sampler, shadow_pos, 2.0).r;
        return (shadow_noise * h_profile) + weather_coverage - 1.0;
    }

    var noise_sample_pos: vec3<f32>;
    if (cloud_layer.is_cirrus) {
        noise_sample_pos = vec3<f32>(coord_along * stretch, coord_perp * 0.4, pos.z * 4.0) * (base_scale * 0.4) + base_time_vec;
    } else {
        let leaning = h_coord * 150.0 * stretch;
        noise_sample_pos = vec3<f32>(coord_along * stretch + leaning, coord_perp, pos.z) * base_scale + base_time_vec;
    }

    let base_noise = textureSampleLevel(base_texture, linear_sampler, noise_sample_pos, 0.0).r;

    var density = base_noise;
    if (cloud_layer.is_cirrus) {
         density = pow(density, 3.5) * 0.5;
    } else if (layer_i < 5u) {
         density = saturate(density * 1.4 - 0.1);
    }

    // Multiply the base density by the profile first for smoother integration
    var cloud_val = (density * h_profile) + weather_coverage - 1.0;
    if cloud_val <= 0.0 { return 0.0; }

    // --- High Frequency Detail ---
    let detail_time = time * WIND_DIRECTION_DETAIL;
    var dw = normalize(WIND_DIRECTION_DETAIL.xy);
    if (length(dw) < 1e-6) { dw = wind; }
    let dperp = vec2<f32>(-dw.y, dw.x);
    let dtime_vec = vec3<f32>(dot(detail_time.xy, dw), dot(detail_time.xy, dperp), detail_time.z);

    let dpos = vec3<f32>(dot(pos.xy, dw) * max(1.0, stretch * 0.5), dot(pos.xy, dperp), pos.z) * DETAIL_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_i] - dtime_vec;
    let detail_noise = textureSampleLevel(details_texture, linear_sampler, dpos, 0.0).r;

    // Smoother erosion masking
    let erosion_mask = select(h_coord, 1.0 - h_coord, layer_i < 5u);
    let erosion = detail_noise * 0.4 * CLOUD_LAYER_DETAILS[layer_i] * (1.1 + erosion_mask);

    cloud_val = saturate(cloud_val - erosion);

    return clamp(cloud_val, 0.0, 1.0);
}
