#define_import_path skybound::clouds
#import skybound::utils::{View, remap, intersect_sphere}

const BASE_SCALE = 0.003;
const BASE_TIME = 0.01;

// Base Parameters
const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0) * 0.2 * BASE_TIME; // Main wind for base shape

// Weather Parameters
const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: vec2<f32> = vec2<f32>(1.0, 0.0) * 0.02 * BASE_TIME; // Weather wind for weather shape

// Detail Parameters
const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(1.0, -1.0, 0.0) * 0.1 * BASE_TIME; // Details move faster

// Curl Parameters
const CURL_NOISE_SCALE: f32 = 0.003 * BASE_SCALE; // Scale for curl noise sampling
const CURL_TIME_SCALE: f32 = 0.1 * BASE_TIME; // Speed of curl noise animation
const CURL_STRENGTH: f32 = 0.2; // Strength of curl distortion

// Cloud scales
const CLOUD_BOTTOM_HEIGHT: f32 = 1000;
const CLOUD_TOP_HEIGHT: f32 = 30000;
const CLOUD_LAYER_HEIGHT: f32 = 1600;
const CLOUD_TOTAL_LAYERS: u32 = 16u;
// The vertical height of the layer
const CLOUD_LAYER_HEIGHTS = array<f32, CLOUD_TOTAL_LAYERS>(
    1200, 1100, 1100, 1200,
    1000, 1100, 1000, 900,
    900, 600, 700, 500,
    300, 300, 300, 250
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
    0.3, 0.15, 0.1, 0.1
);
// Scale stretch on the x axis for the layer
const CLOUD_LAYER_STRETCH = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.5, 2.0,
    2.5, 4.0, 5.0, 6.0
);
// Density multiplier per layer
const CLOUD_LAYER_DENSITIES = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 1.0, 0.8, 0.8,
    0.7, 0.7, 0.5, 0.4,
    0.4, 0.3, 0.25, 0.25,
    0.2, 0.15, 0.15, 0.1
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
    top: f32,
}
fn get_cloud_layer(altitude: f32) -> CloudLayer {
    if altitude <= CLOUD_BOTTOM_HEIGHT || altitude >= CLOUD_TOP_HEIGHT { return CloudLayer(); }

    // Get the index of the layer
    let index: u32 = u32((altitude - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_HEIGHT);
    if index >= CLOUD_TOTAL_LAYERS { return CloudLayer(); }

    // Get the top and bottom height of the layer
    let bottom: f32 = CLOUD_BOTTOM_HEIGHT + f32(index) * CLOUD_LAYER_HEIGHT;
    let top: f32 = bottom + CLOUD_LAYER_HEIGHTS[index];

    // Check if the altitude is within the actual cloud thickness.
    if altitude > top { return CloudLayer(); }

    return CloudLayer(index, bottom, top);
}

fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, base_texture: texture_3d<f32>, details_texture: texture_3d<f32>, motion_texture: texture_2d<f32>, weather_texture: texture_2d<f32>, linear_sampler: sampler) -> f32 {
    // --- Weather Parameters ---
    var cloud_layer = get_cloud_layer(pos.y);
    if cloud_layer.top == 0.0 { return 0.0; }
    let layer_i = cloud_layer.index;
    let weather_pos_2d = (pos.xz + CLOUD_LAYER_OFFSETS[layer_i] * 10000.0) * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;
    let weather_noise = textureSampleLevel(weather_texture, linear_sampler, weather_pos_2d, 0.0);
    let cloud_type_a = weather_noise.b;
    let cloud_type_b = weather_noise.a;
    var weather_coverage = mix(weather_noise.r, weather_noise.g, cloud_type_a) + mix(weather_noise.r, weather_noise.g, cloud_type_b) * 0.5;
    if (weather_coverage <= 0.02) { return 0.0; }

    // --- Height Gradient ---
    let gradient = vec4<f32>(
        cloud_layer.bottom,
        mix(cloud_layer.bottom, cloud_layer.top, 0.2),
        mix(cloud_layer.bottom, cloud_layer.top, 0.6),
        cloud_layer.top
    );
    let height_gradient = smoothstep(gradient.x, gradient.y, pos.y) - smoothstep(gradient.z, gradient.w, pos.y);
    let height_fraction = smoothstep(gradient.x, gradient.w, pos.y);
    // Decide whether the clouds should be weight towards the highest or lowest altitudes
    let layer = f32(layer_i) / f32(CLOUD_TOTAL_LAYERS);
    let cloud_layer_coverage_a = 1.0 - smoothstep(0.0, 0.5, distance(cloud_type_a, layer));
    let cloud_layer_coverage_b = 1.0 - smoothstep(0.0, cloud_type_b * 0.4, distance(1.0, layer));
    weather_coverage *= (cloud_layer_coverage_a + cloud_layer_coverage_b) * (0.5 + pow(layer, 4.0) * 0.5); // Bias the clouds coverage to the highest levels

    // --- Base Cloud Shape ---
    let stretched_scale = vec3<f32>(CLOUD_LAYER_SCALES[layer_i] * CLOUD_LAYER_STRETCH[layer_i], 1.0, CLOUD_LAYER_SCALES[layer_i]);
    let base_scaled_pos = pos * BASE_NOISE_SCALE * stretched_scale + time * WIND_DIRECTION_BASE;
    var base_cloud = textureSampleLevel(base_texture, linear_sampler, vec3<f32>(base_scaled_pos.x, base_scaled_pos.z, base_scaled_pos.y), 0.0).r;
    base_cloud = remap(base_cloud * height_gradient, 1.0 - weather_coverage, 1.0, 0.0, 1.0);
    base_cloud *= weather_coverage;
    if base_cloud <= 0.01 { return 0.0; }

	// --- High Frequency Detail with Curl Distortion ---
    let motion_sample = textureSampleLevel(motion_texture, linear_sampler, pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, 0.0).rgb - 0.5;
    let detail_curl_distortion = motion_sample * CURL_STRENGTH;
    let detail_time_vec = time * WIND_DIRECTION_DETAIL;
    // Strech the detail scale as the verticality increases
    let detail_stretched_scale = mix(CLOUD_LAYER_SCALES[layer_i] * max(CLOUD_LAYER_STRETCH[layer_i] * 0.2, 1.0), 1.0, -0.5);
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE * vec3<f32>(detail_stretched_scale, 1.0, CLOUD_LAYER_SCALES[layer_i]) - detail_time_vec + detail_curl_distortion;

    let detail_noise = textureSampleLevel(details_texture, linear_sampler, vec3<f32>(detail_scaled_pos.x, detail_scaled_pos.z, detail_scaled_pos.y), 0.0).r;
    // Apply more less noise to the lower portion of the cloud, reduced at higher altitudes
    let detail_amount = mix(height_fraction, 1.0, CLOUD_LAYER_DETAILS[layer_i]);
    let hfbm = mix(detail_noise, 1.0 - detail_noise, clamp(detail_amount * 4.0, 0.0, 1.0));
    base_cloud = remap(base_cloud, hfbm * 0.4 * detail_amount, 1.0, 0.0, 1.0);

    return clamp(base_cloud * CLOUD_LAYER_DENSITIES[layer_i], 0.0, 1.0);
}

// Returns vec2(entry_t, exit_t), or vec2(max, 0.0) if no hit
fn clouds_raymarch_entry(ro: vec3<f32>, rd: vec3<f32>, view: View, t_max: f32) -> vec2<f32> {
    let cam_pos = vec3<f32>(0.0, view.planet_radius + ro.y, 0.0);
    let altitude = distance(ro, view.planet_center) - view.planet_radius;

    let bottom_shell_dist = intersect_sphere(cam_pos, rd, view.planet_radius + CLOUD_BOTTOM_HEIGHT);
    let top_shell_dist = intersect_sphere(cam_pos, rd, view.planet_radius + CLOUD_TOP_HEIGHT);

    if altitude >= CLOUD_BOTTOM_HEIGHT && altitude <= CLOUD_TOP_HEIGHT {
        // We are inside the clouds, start raymarching immediately and end when we exit the clouds
        return vec2(0.0, max(bottom_shell_dist, top_shell_dist));
    } else if altitude < CLOUD_BOTTOM_HEIGHT {
        // Below clouds
        if bottom_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
        return vec2<f32>(bottom_shell_dist, select(t_max, top_shell_dist, top_shell_dist > 0.0));
    }
    // We are above the clouds, only raymarch if the intersects the sphere, start at the top_shell_dist and end at bottom_shell_dist
    if top_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
    return vec2<f32>(top_shell_dist, select(t_max, bottom_shell_dist, bottom_shell_dist > 0.0));
}
