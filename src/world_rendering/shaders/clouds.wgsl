#define_import_path skybound::clouds
#import skybound::functions::{remap}

@group(0) @binding(4) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(6) var cloud_motion_texture: texture_2d<f32>;

const COVERAGE: f32 = 0.4; // Overall cloud coverage

// Base Shape
const BASE_NOISE_SCALE: f32 = 0.00005;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(0.001, 0.0, 0.0); // Main wind for base shape

// Detail Shape
const DETAIL_NOISE_SCALE: f32 = 0.003;
const DETAIL_STRENGTH: f32 = 0.3;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(0.008, 0.0, 0.0); // Details move faster

// Motion / Turbulence
const CURL_NOISE_SCALE: f32 = 0.0002;
const CURL_STRENGTH: f32 = 3.0; // How much curl distorts detail sampling
const CURL_TIME_SCALE: f32 = 0.002; // How fast the turbulence itself evolves


/// Sample from the clouds
fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, linear_sampler: sampler) -> f32 {
    let altitude = pos.y;

    // Height gradient
    let low_altitude = smoothstep(1200.0, 1500.0, altitude) * smoothstep(3500.0, 1600.0, altitude);
    let high_altitude = smoothstep(7800.0, 8000.0, altitude) * smoothstep(8500.0, 8000.0, altitude);
    let height_weight = clamp(low_altitude + high_altitude, 0.0, 1.0);
    let coverage = remap(COVERAGE, 0.0, 1.0, 1.0, 0.4);
    if height_weight <= 0.0 { return 0.0; }


    // --- Base Cloud Shape ---
    let base_scaled_pos = pos * BASE_NOISE_SCALE;
    let time_vec = time * WIND_DIRECTION_BASE;

    // Each layer of the FBM gets a different offset in time and from the motion texture.
    let perlin_worley = sample_base(base_scaled_pos + time_vec, linear_sampler).r;
    let worley_fbm_1 = sample_base(base_scaled_pos + time_vec * 0.5, linear_sampler).g;
    let worley_fbm_2 = sample_base(base_scaled_pos + time_vec * 0.25, linear_sampler).b;
    let worley_fbm_3 = sample_base(base_scaled_pos + time_vec * 0.125, linear_sampler).a;

    let worley_fbm = worley_fbm_1 * 0.625 + worley_fbm_2 * 0.25 + worley_fbm_3 * 0.125;
    let base_noise = (abs(perlin_worley) + worley_fbm) * 0.5;

    let base_cloud = remap(base_noise * height_weight, coverage, 1.0, 0.0, 1.0);
    if base_cloud <= 0.0 { return 0.0; }


    // --- High Frequency Detail with Curl Distortion ---
    let motion_sample = sample_motion(pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, linear_sampler).rgb - 0.5;
    let detail_curl_distortion = vec3(motion_sample.r, 0.0, motion_sample.g) * CURL_STRENGTH;
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE + detail_curl_distortion;
    let detail_time_vec = time * WIND_DIRECTION_DETAIL;

    // Sample detail FBM layers. We can give them small offsets too for more dynamism.
    let detail_sample_1 = sample_details(detail_scaled_pos + detail_time_vec, linear_sampler).r;
    let detail_sample_2 = sample_details(detail_scaled_pos + detail_time_vec * 0.5, linear_sampler).g;
    let detail_sample_3 = sample_details(detail_scaled_pos + detail_time_vec * 0.25, linear_sampler).b;
    let detail_fbm = detail_sample_1 * 0.625 + detail_sample_2 * 0.25 + detail_sample_3 * 0.125;

    let whispy_factor = smoothstep(1400.0, 1600.0, altitude) * (1.0 - smoothstep(1600.0, 1800.0, altitude));
    let final_detail_noise = mix(detail_fbm, 1.0 - detail_fbm, whispy_factor);

    let eroded_cloud = base_cloud - final_detail_noise * DETAIL_STRENGTH;
    return clamp(eroded_cloud, 0.0, 1.0);
}

fn sample_base(pos: vec3<f32>, linear_sampler: sampler) -> vec4<f32> {
    return textureSample(cloud_base_texture, linear_sampler, vec3<f32>(pos.x, pos.z, pos.y));
}

fn sample_details(pos: vec3<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(cloud_details_texture, linear_sampler, vec3<f32>(pos.x, pos.z, pos.y)).xyz;
}

fn sample_motion(pos: vec2<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(cloud_motion_texture, linear_sampler, pos).xyz;
}
