#define_import_path skybound::clouds
#import skybound::functions::{remap}
#import skybound::sky::render_sky

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

// Cloud scales
const CLOUDS_BOTTOM_HEIGHT: f32 = 1500.0;
const CLOUDS_TOP_HEIGHT: f32 = 40000.0;


// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 12.0;
const STEP_SIZE_OUTSIDE: f32 = 24.0;

const STEP_DISTANCE_SCALING_START: f32 = 500.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.001; // How much to scale step size by distance, larger means larger steps

const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 30.0;
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));

// Lighting Parameters
const AMBIENT_AUR_COLOR: vec3<f32> = vec3(0.4, 0.1, 0.6);
const DENSITY: f32 = 0.05; // Base density for lighting
const SILVER_SPREAD: f32 = 0.1;
const SILVER_INTENSITY: f32 = 1.0;

const FADE_START_DISTANCE: f32 = 1000.0;
const FADE_END_DISTANCE: f32 = 200000.0;


fn get_height_fraction(altitude: f32) -> f32 {
	return clamp((altitude - CLOUDS_BOTTOM_HEIGHT) / (CLOUDS_TOP_HEIGHT - CLOUDS_BOTTOM_HEIGHT), 0.0, 1.0);
}

const K: f32 = 0.0795774715459;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
	return K * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

const PLANE_NORMAL: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
fn intersect_cloud_start(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let denom = dot(PLANE_NORMAL, rd);
    if abs(denom) < 1e-6 {
        return -1.0; // Parallel
    }
    let t = -(dot(PLANE_NORMAL, ro) - CLOUDS_BOTTOM_HEIGHT) / denom;
    return t;
}

fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32, fast: bool, linear_sampler: sampler) -> f32 {
    let altitude = pos.y;

    // --- Height Gradient ---
    let gradient_low = vec4<f32>(1500.0, 1650.0, 2500.0, 3000.0);
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
    let time_vec = time * WIND_DIRECTION_BASE;
    let base_scaled_pos = pos * BASE_NOISE_SCALE + time_vec;

    let base_noise = sample_base(base_scaled_pos, linear_sampler);
	let fbm = base_noise.g * 0.625 + base_noise.b * 0.25 + base_noise.a * 0.125;
	var base_cloud = remap(base_noise.r, -(1.0 - fbm), 1.0, 0.0, 1.0);
	base_cloud = remap(base_cloud * height_gradient, 1.0 - COVERAGE, 1.0, 0.0, 1.0);
	base_cloud *= COVERAGE;

	if base_cloud <= 0.0 { return 0.0; }

	// --- High Frequency Detail with Curl Distortion (TODO) ---
	// if !fast {
    	// let motion_sample = sample_motion(pos.xz * CURL_NOISE_SCALE + time * CURL_TIME_SCALE, linear_sampler).rgb - 0.5;
        // let detail_curl_distortion = vec3(motion_sample.r, 0.0, motion_sample.g) * CURL_STRENGTH;
        let detail_time_vec = time * WIND_DIRECTION_DETAIL;
        let detail_scaled_pos = pos * DETAIL_NOISE_SCALE - detail_time_vec;

    	let detail_noise = sample_details(detail_scaled_pos, linear_sampler);
    	var hfbm = detail_noise.r * 0.625 + detail_noise.g * 0.25 + detail_noise.b * 0.125;
    	hfbm = mix(hfbm, 1.0 - hfbm, clamp(height_fraction * 4.0, 0.0, 1.0));
    	base_cloud = remap(base_cloud, hfbm * 0.4 * height_fraction, 1.0, 0.0, 1.0);
	// }

	return clamp(base_cloud, 0.0, 1.0);
}

fn render_clouds(ro: vec3<f32>, rd: vec3<f32>, sun_dir: vec3<f32>, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> vec4<f32> {
    // Determine raymarch start and end distances
    var t = dither * STEP_SIZE_INSIDE;
    var t_end = t_max;
    let start_intersect = intersect_cloud_start(ro, rd);
    if (ro.y < CLOUDS_BOTTOM_HEIGHT) {
        // Camera below clouds, start raymarching at intersection if valid
        t += max(start_intersect, 0.0);
        t_end = min(t_end, FADE_END_DISTANCE);
    } else {
        // Camera above clouds, start at 0 and end at intersection if valid
        t_end = min(t_end, select(FADE_END_DISTANCE, start_intersect, start_intersect > 0.0));
    }
    if t >= t_end {
        return vec4<f32>(0.0);
    }

	// Precalculate sun, sky and ambient colors
	let sky_col = render_sky(rd, sun_dir, ro.y);
	let atmosphere_sun = render_sky(sun_dir, sun_dir, ro.y) * 0.1;
	let atmosphere_ambient = render_sky(normalize(vec3<f32>(1.0, 1.0, 0.0)), sun_dir, ro.y);
	let atmosphere_ground = AMBIENT_AUR_COLOR * 100.0;

    // Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.6);
    let hg_silver = henyey_greenstein(cos_theta, 0.99 - SILVER_SPREAD) * SILVER_INTENSITY;
    let hg_back = henyey_greenstein(cos_theta, -0.1);
    let phase = max(hg_forward, max(hg_silver, hg_back)) + 0.1;

    // Accumulation variables
    var acc_color = vec3(0.0);
    var acc_alpha = 0.0;
    var transmittance = 1.0;
    var steps_outside_cloud = 0;

    // Start raymarching
    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= t_end || acc_alpha >= ALPHA_THRESHOLD || t >= FADE_END_DISTANCE {
            break;
        }

        // Scale step size based on distance from camera
        var step_scaler = 1.0;
        if t > STEP_DISTANCE_SCALING_START {
            step_scaler = 1.0 + min((t - STEP_DISTANCE_SCALING_START) * STEP_DISTANCE_SCALING_FACTOR, 16.0);
        }
        // Reduce scaling when close to surfaces
        let close_threshold = STEP_SIZE_OUTSIDE * step_scaler;
        let distance_left = t_max - t;
        if distance_left < close_threshold {
            let norm = clamp(distance_left / close_threshold, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }
        var step = STEP_SIZE_OUTSIDE * step_scaler;

        // Sample the cloud
        let pos = ro + rd * (t + dither * step);
        let step_density = sample_clouds(pos, t, time, false, linear_sampler);

        // Adjust t to effectively "backtrack" and take smaller steps when entering a cloud
        if step_density > 0.0 {
            if steps_outside_cloud != 0 {
                // First step into the cloud;
                steps_outside_cloud = 0;
                t = max(t + (-STEP_SIZE_OUTSIDE + STEP_SIZE_INSIDE) * step_scaler, 0.0);
                continue;
            }
        } else {
            steps_outside_cloud += 1;
        }

        if step_density > 0.0 {
            step = STEP_SIZE_INSIDE * step_scaler;

            let step_transmittance = exp(-DENSITY * step_density * step);
            let alpha_step = (1.0 - step_transmittance);

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var lightmarch_pos = pos;
            for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
                lightmarch_pos += (sun_dir + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE;
                density_sunwards += sample_clouds(lightmarch_pos, t, time, true, linear_sampler);
            }
            // Take a single distant sample
            lightmarch_pos += sun_dir * LIGHT_STEP_SIZE * 18.0;
            let lheight_fraction = get_height_fraction(lightmarch_pos.y);
            density_sunwards += pow(sample_clouds(lightmarch_pos, t, time, true, linear_sampler), (1.0 - lheight_fraction) * 0.8 + 0.5);

            // Captures the direct lighting from the sun
			let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE);
			let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE * 0.25) * 0.7;
			let beers_total = max(beers, beers2);

			// Compute in-scattering
			let height_fraction = get_height_fraction(pos.y);
			let aur_ambient = mix(atmosphere_ground, vec3(1.0), pow(height_fraction, 0.15));
            let ambient = aur_ambient * DENSITY * mix(atmosphere_ambient, vec3(1.0), 0.4) * (sun_dir.y);
            let in_scattering = ambient + beers_total * atmosphere_sun * phase;

            // Compute emission, aur color if low altitude
            let emission = AMBIENT_AUR_COLOR * max((1.0 - height_fraction) - 0.5, 0.0) * 0.0005 * step;

			acc_alpha += alpha_step * (1.0 - acc_alpha);
			acc_color += in_scattering * transmittance * alpha_step + emission;

			transmittance *= step_transmittance;
        }

        t += step;
    }

    acc_alpha = min(acc_alpha * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    return clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
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
