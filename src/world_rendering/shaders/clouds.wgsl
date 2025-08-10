#define_import_path skybound::clouds
#import skybound::functions::{remap, intersect_sphere}
#import skybound::sky::AtmosphereData

@group(0) @binding(4) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(6) var cloud_motion_texture: texture_2d<f32>;
@group(0) @binding(7) var cloud_weather_texture: texture_2d<f32>;

// Base Parameters
const BASE_NOISE_SCALE: f32 = 0.00008;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0) * 0.005; // Main wind for base shape

// Weather Parameters
const WEATHER_NOISE_SCALE: f32 = 0.000006;
const WIND_DIRECTION_WEATHER: vec2<f32> = vec2<f32>(1.0, 0.0) * 0.0005; // Weather wind for weather shape

// Detail Parameters
const DETAIL_NOISE_SCALE: f32 = 0.001;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(0.3, -0.3, 0.0) * 0.1; // Details move faster

// Curl Parameters
const CURL_NOISE_SCALE: f32 = 0.000003;  // Scale for curl noise sampling
const CURL_TIME_SCALE: f32 = 0.0004;    // Speed of curl noise animation
const CURL_STRENGTH: f32 = 8.0;      // Strength of curl distortion

// Cloud scales
const CLOUDS_BOTTOM_HEIGHT: f32 = 1500.0;
const CLOUDS_TOP_HEIGHT: f32 = 40000.0;


fn get_height_fraction(altitude: f32) -> f32 {
    return clamp((altitude - CLOUDS_BOTTOM_HEIGHT) / (CLOUDS_TOP_HEIGHT - CLOUDS_BOTTOM_HEIGHT), 0.0, 1.0);
}

fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, linear_sampler: sampler) -> f32 {
    let altitude = pos.y;

    // --- Height Gradient ---
    let gradient_low = vec4<f32>(1500.0, 1650.0, 2250.0, 3000.0);
    let gradient_high = vec4<f32>(6500.0, 6650.0, 7000.0, 7500.0);
    var gradient = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if altitude >= gradient_low.x && altitude <= gradient_low.w {
        gradient = gradient_low;
    } else if altitude >= gradient_high.x && altitude <= gradient_high.w {
        gradient = gradient_high;
    } else {
        return 0.0;
    }
    let height_gradient = smoothstep(gradient.x, gradient.y, altitude) - smoothstep(gradient.z, gradient.w, altitude);
    let height_fraction = smoothstep(gradient.x, gradient.w, altitude);

    // --- Base Cloud Shape ---
    let base_scaled_pos = pos * BASE_NOISE_SCALE + time * WIND_DIRECTION_BASE;
    let weather_pos = pos.xz * WEATHER_NOISE_SCALE + time * WIND_DIRECTION_WEATHER;

    let base_noise = sample_base(base_scaled_pos, linear_sampler);
    let fbm = base_noise.g * 0.625 + base_noise.b * 0.25 + base_noise.a * 0.125;
    var base_cloud = remap(base_noise.r, -(1.0 - fbm), 1.0, 0.0, 1.0);

    let weather_noise = sample_weather(weather_pos, linear_sampler);
    let weather_coverage = remap(pow(weather_noise.r, 0.5), 0.0, 1.0, 0.0, 0.5);

    base_cloud = remap(base_cloud * height_gradient, 1.0 - weather_coverage, 1.0, 0.0, 1.0);
    base_cloud *= weather_noise.r;

    if base_cloud <= 0.0 { return 0.0; }

	// --- High Frequency Detail with Curl Distortion ---
    let motion_sample = sample_motion(pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, linear_sampler).rgb - 0.5;
    let detail_curl_distortion = motion_sample * CURL_STRENGTH;
    let detail_time_vec = time * WIND_DIRECTION_DETAIL;
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE - detail_time_vec + detail_curl_distortion;

    let detail_noise = sample_details(detail_scaled_pos, linear_sampler);
    var hfbm = detail_noise.r * 0.625 + detail_noise.g * 0.25 + detail_noise.b * 0.125;
    hfbm = mix(hfbm, 1.0 - hfbm, clamp(height_fraction * 4.0, 0.0, 1.0));
    base_cloud = remap(base_cloud, hfbm * 0.4 * height_fraction, 1.0, 0.0, 1.0);

    return clamp(base_cloud, 0.0, 1.0);
}

// Returns vec2(entry_t, exit_t), or vec2(max, 0.0) if no hit
fn clouds_raymarch_entry(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, t_max: f32) -> vec2<f32> {
    let cam_pos = vec3<f32>(0.0, atmosphere.planet_radius + ro.y, 0.0);
    let altitude = distance(ro, atmosphere.planet_center) - atmosphere.planet_radius;

    let bottom_shell_dist = intersect_sphere(cam_pos, rd, atmosphere.planet_radius + CLOUDS_BOTTOM_HEIGHT);
    let top_shell_dist = intersect_sphere(cam_pos, rd, atmosphere.planet_radius + CLOUDS_TOP_HEIGHT);

    var t_start: f32;
    var t_end: f32;
    if altitude >= CLOUDS_BOTTOM_HEIGHT && altitude <= CLOUDS_TOP_HEIGHT {
        // We are inside the clouds, start raymarching immediately and end when we exit the clouds
        t_start = 0.0;
        t_end = max(bottom_shell_dist, top_shell_dist); // Whichever is further along the ray
    } else if altitude < CLOUDS_BOTTOM_HEIGHT {
        // Below clouds
        if bottom_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
        t_start = bottom_shell_dist;
        if top_shell_dist > 0.0 {
            t_end = top_shell_dist;
        } else {
            t_end = t_max;
        }
    } else {
        // We are above the clouds, only raymarch if the intersects the sphere, start at the top_shell_dist and end at bottom_shell_dist
        if top_shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
        t_start = top_shell_dist;
        if bottom_shell_dist > 0.0 {
            t_end = bottom_shell_dist;
        } else {
            t_end = t_max;
        }
    }

    return vec2<f32>(t_start, t_end);
}

fn sample_base(pos: vec3<f32>, linear_sampler: sampler) -> vec4<f32> {
    return textureSample(cloud_base_texture, linear_sampler, vec3<f32>(pos.x, pos.z, pos.y));
}

fn sample_details(pos: vec3<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(cloud_details_texture, linear_sampler, vec3<f32>(pos.x, pos.z, pos.y)).rgb;
}

fn sample_motion(pos: vec2<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(cloud_motion_texture, linear_sampler, pos).rgb;
}

fn sample_weather(pos: vec2<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(cloud_weather_texture, linear_sampler, pos).rgb;
}
