#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::functions::{remap, blue_noise}
#import skybound::clouds::{sample_clouds, get_height_fraction}
#import skybound::aur_fog::sample_fog
#import skybound::sky::render_sky
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
struct View {
    world_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    planet_rotation: vec4<f32>,
    latitude: f32,
    longitude: f32,
    latitude_meters: f32,
    longitude_meters: f32,
    altitude: f32,
};
@group(0) @binding(1) var<uniform> globals: Globals;
struct Globals {
    time: f32,
    planet_radius: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
}
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

@group(0) @binding(4) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(6) var cloud_motion_texture: texture_2d<f32>;
@group(0) @binding(7) var fog_noise_texture: texture_3d<f32>;

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 8.0;
const STEP_SIZE_OUTSIDE: f32 = 18.0;

const STEP_DISTANCE_SCALING_START: f32 = 500.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.001; // How much to scale step size by distance, larger means larger steps

const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 30.0;
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));

// Lighting Parameters
const MIN_SUN_DOT: f32 = sin(radians(-8.0)); // How far below the horizon before the switching to aur light
const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const AMBIENT_AUR_COLOR: vec3<f32> = vec3(0.4, 0.1, 0.6);
const DENSITY: f32 = 0.05; // Base density for lighting
const SILVER_SPREAD: f32 = 0.1;
const SILVER_INTENSITY: f32 = 1.0;

const FOG_START_DISTANCE: f32 = 1000.0;
const FOG_END_DISTANCE: f32 = 200000.0;

// Cloud density and colour
struct CloudSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume(pos: vec3<f32>, dist: f32, fast: bool) -> CloudSample {
    var sample: CloudSample;

    // Thick aur fog below 0m
    if !fast {
        let fog_sample = sample_fog(pos, dist, globals.time, linear_sampler);
        sample.density = fog_sample.contribution;
        sample.color = fog_sample.color;
        sample.emission = fog_sample.emission;
    }

    // Sample our clouds
    let clouds_sample = sample_clouds(pos, dist, globals.time, fast, linear_sampler);
    sample.density += clouds_sample;
    if clouds_sample > 0.0 {
        sample.color = vec3(1.0);
    }

    return sample;
}

// Lighting Functions
const K: f32 = 0.0795774715459;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
	return K * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

// Raymarch through all the clouds
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;

    // Load depth and unproject to clip space
    let depth = textureSample(depth_texture, linear_sampler, uv);
    let ndc = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);

    // Reconstruct world‑space pos
    let world_pos4 = view.world_from_clip * ndc;
    let world_pos3 = world_pos4.xyz / world_pos4.w;

    // Ray origin & dir
    let ro = view.world_position; //vec3(view.longitude_meters, view.altitude, -view.latitude_meters);
    let rd_vec = world_pos3 - ro;
    let t_max = length(rd_vec);
    let rd = normalize(rd_vec);

    let dither = fract(blue_noise(pix));

    var acc_color = vec3(0.0);
    var acc_alpha = 0.0;
    var transmittance = 1.0;
    var t = dither * STEP_SIZE_INSIDE;
    var steps_outside_cloud = 0;

    // Get sun direction and intensity, mix between aur light (straight up) and sun
    let sun_dot = globals.sun_direction.y;
    let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    let sun_dir = normalize(mix(AUR_DIR, globals.sun_direction, sun_t));

	// Precalculate sun, sky and ambient colors
	let sky_col = render_sky(rd, sun_dir, view.altitude);
	let atmosphere_sun = render_sky(sun_dir, sun_dir, view.altitude) * 0.1;
	let atmosphere_ambient = render_sky(normalize(vec3<f32>(1.0, 1.0, 0.0)), sun_dir, view.altitude);
	// let atmosphere_ground = render_sky(normalize(vec3<f32>(1.0, -1.0, 0.0)), sun_dir, view.altitude);
	let atmosphere_ground = AMBIENT_AUR_COLOR * 100.0;

    // Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.6);
    let hg_silver = henyey_greenstein(cos_theta, 0.99 - SILVER_SPREAD) * SILVER_INTENSITY;
    let hg_back = henyey_greenstein(cos_theta, -0.1);
    let phase = max(hg_forward, max(hg_silver, hg_back)) + 0.1;

    // Render out the world poles
    let pole_color = render_poles(ro, rd, view.planet_rotation, globals.planet_radius);

    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= t_max || acc_alpha >= ALPHA_THRESHOLD || t >= FOG_END_DISTANCE {
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
        let cloud_sample = sample_volume(pos, t, false);
        let step_density = cloud_sample.density;

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
            transmittance *= step_transmittance;

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var lightmarch_pos = pos;
            for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
                lightmarch_pos += (sun_dir + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE;
                density_sunwards += sample_volume(lightmarch_pos, t, true).density;
            }
            // Take a single distant sample
            lightmarch_pos += sun_dir * LIGHT_STEP_SIZE * 18.0;
            let lheight_fraction = get_height_fraction(lightmarch_pos.y);
            density_sunwards += pow(sample_volume(lightmarch_pos, t, true).density, (1.0 - lheight_fraction) * 0.8 + 0.5);

            // Captures the direct lighting from the sun
			let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE);
			let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE * 0.25) * 0.7;
			let beers_total = max(beers, beers2);

			// Compute in-scattering
			let height_fraction = get_height_fraction(pos.y);
			let aur_ambient = mix(atmosphere_ground, vec3(1.0), pow(height_fraction, 0.15));
            let ambient = aur_ambient * DENSITY * mix(atmosphere_ambient, vec3(1.0), 0.4) * (sun_dir.y);
            let in_scattering = (ambient + beers_total * atmosphere_sun * phase) * cloud_sample.color;

            // Compute emission, using clouds emission then adding aur color if low altitude
            let aur_emission = AMBIENT_AUR_COLOR * max((1.0 - height_fraction) - 0.5, 0.0) * 0.0005;
            let emission = (cloud_sample.emission + aur_emission) * step;

			let alpha_step = (1.0 - step_transmittance);
			acc_alpha += alpha_step * (1.0 - acc_alpha);
			acc_color += in_scattering * transmittance * alpha_step + emission;
        }

        t += step;
    }

    acc_alpha = min(acc_alpha * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    if depth <= 0.00001 {
        // Blend in poles behind clouds
        acc_color += pole_color.rgb * pole_color.a * (1.0 - acc_alpha);
        acc_alpha += pole_color.a * (1.0 - acc_alpha);

        // Add our sky in the background
        acc_color += vec3(sky_col * (1.0 - acc_alpha));
        acc_alpha = 1.0;
    }

    return clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
}
