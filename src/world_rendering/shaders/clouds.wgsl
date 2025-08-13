#define_import_path skybound::clouds
#import skybound::utils::{View, remap, intersect_sphere}

const BASE_SCALE = 0.003;
const BASE_TIME = 0.01;

// Base Parameters
const BASE_NOISE_SCALE: f32 = 0.03 * BASE_SCALE;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0) * 0.2 * BASE_TIME; // Main wind for base shape

// Weather Parameters
const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: vec2<f32> = vec2<f32>(1.0, 0.0) * 0.02 * BASE_TIME; // Weather wind for weather shape

// Detail Parameters
const DETAIL_NOISE_SCALE: f32 = 0.3 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(1.0, -1.0, 0.0) * 0.1 * BASE_TIME; // Details move faster

// Curl Parameters
const CURL_NOISE_SCALE: f32 = 0.0005 * BASE_SCALE; // Scale for curl noise sampling
const CURL_TIME_SCALE: f32 = 0.005 * BASE_TIME; // Speed of curl noise animation
const CURL_STRENGTH: f32 = 6.0; // Strength of curl distortion

// Cloud scales
const CLOUDS_BOTTOM_HEIGHT: f32 = 1000.0;
const CLOUDS_TOP_HEIGHT: f32 = 40000.0;
const CLOUD_TOTAL_LAYERS: u32 = 16u;
const CLOUD_LAYER_HEIGHTS = array<u32, CLOUD_TOTAL_LAYERS>(
    3500, 3000, 2500, 3000,
    2000, 1500, 1500, 2000,
    1500, 1000, 1000, 800,
    600, 800, 800, 1000
);
const CLOUD_LAYER_SCALES = array<f32, CLOUD_TOTAL_LAYERS>(
    1.0, 0.9, 0.8, 0.8,
    0.7, 0.7, 0.6, 0.6,
    0.5, 0.5, 0.5, 0.4,
    0.4, 0.3, 0.2, 0.1
);


fn get_height_fraction(altitude: f32) -> f32 {
    return clamp((altitude - CLOUDS_BOTTOM_HEIGHT) / (CLOUDS_TOP_HEIGHT - CLOUDS_BOTTOM_HEIGHT), 0.0, 1.0);
}

// Gets the current cloud layers index, bottom and top height
struct CloudLayer {
    index: f32,
    layer: f32,
    bottom: f32,
    top: f32,
    scale: f32
}
fn get_cloud_layer(altitude: f32) -> CloudLayer {
    var bottom: f32 = CLOUDS_BOTTOM_HEIGHT;
    for (var i = 0u; i < CLOUD_TOTAL_LAYERS; i++) {
        let top = bottom + f32(CLOUD_LAYER_HEIGHTS[i]);
        if (altitude >= bottom && altitude <= top) {
            return CloudLayer(f32(i), f32(i) / f32(CLOUD_TOTAL_LAYERS), bottom, top, CLOUD_LAYER_SCALES[i]);
        }
        bottom = top + 500.0;
    }
    return CloudLayer(); // Altitude out of range
}

fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, simple: bool, base_texture: texture_3d<f32>, details_texture: texture_3d<f32>, motion_texture: texture_2d<f32>, weather_texture: texture_2d<f32>, linear_sampler: sampler) -> f32 {
    // --- Weather Parameters ---
    var cloud_layer = get_cloud_layer(pos.y);
    let weather_pos_2d = pos.xz * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;
    let weather_noise = textureSampleLevel(weather_texture, linear_sampler, weather_pos_2d, 0.0);
    let cloud_type_a = weather_noise.b;
    let cloud_type_b = weather_noise.a;
    var weather_coverage = mix(weather_noise.r, weather_noise.g, cloud_type_a) + mix(weather_noise.r, weather_noise.g, cloud_type_b) * 0.5;
    if weather_coverage <= 0.0 { return 0.0; }

    // --- Height Gradient ---
    if cloud_layer.top == 0.0 { return 0.0; }
    let gradient = vec4<f32>(
        cloud_layer.bottom,
        mix(cloud_layer.bottom, cloud_layer.top, 0.1),
        mix(cloud_layer.bottom, cloud_layer.top, 0.2),
        cloud_layer.top
    );
    let height_gradient = smoothstep(gradient.x, gradient.y, pos.y) - smoothstep(gradient.z, gradient.w, pos.y);
    let height_fraction = smoothstep(gradient.x, gradient.w, pos.y);
    // Decide whether the clouds should be weight towards the highest or lowest altitudes
    let cloud_layer_coverage_a = 1.0 - smoothstep(0.0, 0.6, distance(cloud_type_a, cloud_layer.layer));
    let cloud_layer_coverage_b = 1.0 - smoothstep(0.0, 0.4, distance(cloud_type_b, cloud_layer.layer));
    weather_coverage *= (cloud_layer_coverage_a + cloud_layer_coverage_b) * 0.5;

    // --- Base Cloud Shape ---
    let base_scaled_pos = pos * BASE_NOISE_SCALE * cloud_layer.scale + time * WIND_DIRECTION_BASE;
    var base_cloud = textureSampleLevel(base_texture, linear_sampler, vec3<f32>(base_scaled_pos.x, base_scaled_pos.z, base_scaled_pos.y), 0.0).r;

    base_cloud = remap(base_cloud * height_gradient, 1.0 - weather_coverage, 1.0, 0.0, 1.0);
    base_cloud *= weather_coverage;
    if base_cloud <= 0.0 { return 0.0; }
    if simple { return base_cloud; }

	// --- High Frequency Detail with Curl Distortion ---
    let motion_sample = textureSampleLevel(motion_texture, linear_sampler, pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, 0.0).rgb - 0.5;
    let detail_curl_distortion = motion_sample * CURL_STRENGTH;
    let detail_time_vec = time * WIND_DIRECTION_DETAIL;
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE - detail_time_vec + detail_curl_distortion;

    let detail_noise = textureSampleLevel(details_texture, linear_sampler, vec3<f32>(detail_scaled_pos.x, detail_scaled_pos.z, detail_scaled_pos.y), 0.0).r;
    let hfbm = mix(detail_noise, 1.0 - detail_noise, clamp(height_fraction * 4.0, 0.0, 1.0));
    base_cloud = remap(base_cloud, hfbm * 0.4 * height_fraction, 1.0, 0.0, 1.0);

    return clamp(base_cloud, 0.0, 1.0);
}

// Returns vec2(entry_t, exit_t), or vec2(max, 0.0) if no hit
fn clouds_raymarch_entry(ro: vec3<f32>, rd: vec3<f32>, view: View, t_max: f32) -> vec2<f32> {
    let cam_pos = vec3<f32>(0.0, view.planet_radius + ro.y, 0.0);
    let altitude = distance(ro, view.planet_center) - view.planet_radius;

    let bottom_shell_dist = intersect_sphere(cam_pos, rd, view.planet_radius + CLOUDS_BOTTOM_HEIGHT);
    let top_shell_dist = intersect_sphere(cam_pos, rd, view.planet_radius + CLOUDS_TOP_HEIGHT);

    if altitude >= CLOUDS_BOTTOM_HEIGHT && altitude <= CLOUDS_TOP_HEIGHT {
        // We are inside the clouds, start raymarching immediately and end when we exit the clouds
        return vec2(0.0, max(bottom_shell_dist, top_shell_dist));
    } else if altitude < CLOUDS_BOTTOM_HEIGHT {
        // Below clouds
        if bottom_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
        return vec2<f32>(bottom_shell_dist, select(t_max, top_shell_dist, top_shell_dist > 0.0));
    }
    // We are above the clouds, only raymarch if the intersects the sphere, start at the top_shell_dist and end at bottom_shell_dist
    if top_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
    return vec2<f32>(top_shell_dist, select(t_max, bottom_shell_dist, bottom_shell_dist > 0.0));
}
