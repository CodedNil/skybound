#define_import_path skybound::clouds
#import skybound::functions::{remap}

@group(0) @binding(4) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(6) var cloud_motion_texture: texture_2d<f32>;

const COVERAGE: f32 = 0.25; // Overall cloud coverage

// Base Shape
const BASE_NOISE_SCALE: f32 = 0.00008;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(0.001, 0.0, 0.0); // Main wind for base shape

// Detail Shape
const DETAIL_NOISE_SCALE: f32 = 0.001;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(0.008, -0.008, 0.0); // Details move faster


/// Sample from the clouds
fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, linear_sampler: sampler) -> f32 {
    let altitude = pos.y;

    // Height gradient
    let gradient_low = vec4<f32>(1500.0, 1650.0, 2500, 3000.0);
    let gradient_high = vec4<f32>(6500.0, 6650.0, 7000, 7500.0);
    let height_gradient = smoothstep(gradient_low.x, gradient_low.y, altitude) - smoothstep(gradient_low.z, gradient_low.w, altitude) + smoothstep(gradient_high.x, gradient_high.y, altitude) - smoothstep(gradient_high.z, gradient_high.w, altitude);
    let coverage = remap(COVERAGE, 0.0, 1.0, 1.0, 0.4);
    if height_gradient <= 0.0 { return 0.0; }

    // --- Base Cloud Shape ---
    let time_vec = time * WIND_DIRECTION_BASE;
    let base_scaled_pos = pos * BASE_NOISE_SCALE + time_vec;

    let base_noise = sample_base(base_scaled_pos, linear_sampler);
	let fbm = base_noise.g * 0.625 + base_noise.b * 0.25 + base_noise.a * 0.125;
	var base_cloud = remap(base_noise.r, -(1.0 - fbm), 1.0, 0.0, 1.0);
	base_cloud = remap(base_cloud * height_gradient, 1.0 - COVERAGE, 1.0, 0.0, 1.0);
	base_cloud *= COVERAGE;

	// --- High Frequency Detail with Curl Distortion (TODO) ---
	// let motion_sample = sample_motion(pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, linear_sampler).rgb - 0.5;
    // let detail_curl_distortion = vec3(motion_sample.r, 0.0, motion_sample.g) * CURL_STRENGTH;
    let detail_time_vec = time * WIND_DIRECTION_DETAIL;
    let detail_scaled_pos = pos * DETAIL_NOISE_SCALE - detail_time_vec;
    let height_fraction = smoothstep(1500.0, 7000.0, altitude); // How high through the total cloud height we are

	let detail_noise = sample_details(detail_scaled_pos, linear_sampler);
	var hfbm = detail_noise.r * 0.625 + detail_noise.g * 0.25 + detail_noise.b * 0.125;
	hfbm = mix(hfbm, 1.0 - hfbm, clamp(height_fraction * 4.0, 0.0, 1.0));
	base_cloud = remap(base_cloud, hfbm * 0.4 * height_fraction, 1.0, 0.0, 1.0);
	return pow(clamp(base_cloud, 0.0, 1.0), (1.0 - height_fraction) * 0.8 + 0.5);
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
