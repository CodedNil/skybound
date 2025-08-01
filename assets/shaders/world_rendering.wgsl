#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::aur_fog::sample_fog
#import skybound::functions::{fbm_3, blue_noise}
#import skybound::poles::{render_poles}

@group(0) @binding(0) var<uniform> view: View;
struct View {
    world_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    planet_rotation: vec4<f32>,
    latitude: f32,
    longitude: f32,
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

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 4.0;
const STEP_SIZE_OUTSIDE: f32 = 12.0;

const STEP_DISTANCE_SCALING_START: f32 = 100.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.0005; // How much to scale step size by distance

const LIGHT_STEPS: i32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 3.0;

// Lighting Parameters
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);
const MIN_SUN_DOT: f32 = sin(radians(-8.0)); // How far below the horizon before the switching to aur light
const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const AUR_COLOR: vec3<f32> = vec3(0.3, 0.2, 0.8) * 0.15;
const AMBIENT_COLOR: vec3<f32> = vec3(0.7, 0.8, 1.0) * 0.5;
const AMBIENT_AUR_COLOR: vec3<f32> = AUR_COLOR * 0.2;

const FOG_START_DISTANCE: f32 = 1000.0;
const FOG_END_DISTANCE: f32 = 100000.0;

const SHADOW_EXTINCTION: f32 = 2.0; // Higher = deeper core shadows

// Cloud Material Parameters
const BACK_SCATTERING: f32 = 1.0; // Backscattering
const BACK_SCATTERING_FALLOFF: f32 = 30.0; // Backscattering falloff
const OMNI_SCATTERING: f32 = 0.8; // Omnidirectional Scattering
const TRANSMISSION_SCATTERING: f32 = 1.0; // Transmission Scattering
const TRANSMISSION_FALLOFF: f32 = 2.0; // Transmission falloff
const BASE_TRANSMISSION: f32 = 0.1; // Light that doesn't get scattered at all

// Rayleigh Scattering Parameters
const PI: f32 = 3.14159265;
const K_RAYLEIGH: f32 = 0.005;
const K_R4PI: f32 = K_RAYLEIGH * 4.0 * PI;
const K_LAMBDA: vec3<f32> = vec3(0.65, 0.57, 0.475); // Wavelength factors
const K_SCALE_DEPTH: f32 = 0.25;

// Cloud density and colour
struct CloudSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_cloud(pos: vec3<f32>, dist: f32) -> CloudSample {
    var sample: CloudSample;
    let altitude = pos.y;

    // Thick aur fog below 0m
    let fog_sample = sample_fog(pos, dist, globals.time);
    sample.emission = fog_sample.emission;

    // Low clouds starting at y=200
    let low_gradient = smoothstep(800.0, 1200.0, altitude) * smoothstep(6000.0, 5000.0, altitude);
    var cloud_contribution = 0.0;

    // if low_gradient > 0.01 {
    //     let base_noise = fbm_3(pos * 0.001 + vec3(0.0, globals.time * 0.02, 0.0), 6) - 0.2;
    //     cloud_contribution = clamp(base_noise * 0.1, 0.0, 1.0) * low_gradient;
    // }

    sample.density = fog_sample.contribution + cloud_contribution;

    if sample.density > 0.0 {
        sample.color = (fog_sample.color * fog_sample.contribution + cloud_contribution) / sample.density;
    } else {
        sample.color = vec3(1.0);
    }

    return sample;
}

// Lighting Functions
fn beer(material_amount: f32) -> f32 {
    return exp(-material_amount);
}

fn transmission(light: vec3<f32>, material_amount: f32) -> vec3<f32> {
    return beer(material_amount * (1.0 - BASE_TRANSMISSION)) * light;
}

fn light_scattering(light: vec3<f32>, angle: f32) -> vec3<f32> {
    var a = (angle + 1.0) * 0.5; // Angle between 0 and 1

    var ratio = 0.0;
    ratio += BACK_SCATTERING * pow(1.0 - a, BACK_SCATTERING_FALLOFF);
    ratio += TRANSMISSION_SCATTERING * pow(a, TRANSMISSION_FALLOFF);
    ratio = ratio * (1.0 - OMNI_SCATTERING) + OMNI_SCATTERING;

    return light * ratio * (1.0 - BASE_TRANSMISSION);
}

// Simple sky shading
fn rayleigh_phase(cos2: f32) -> f32 {
    return 0.75 * (1.0 + cos2);
}

fn sky_color(rd: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_dot = sun_dir.y; // Sun elevation
    let elevation = rd.y; // Ray elevation, normalized
    let view_sun_dot = dot(rd, sun_dir); // Cosine of angle between view and sun
    let view_sun_dot2 = view_sun_dot * view_sun_dot; // For phase function
    let t = (elevation + 1.0) / 2.0; // Gradient from horizon to zenith [0, 1]

    // Base colors
    let day_horizon = vec3(0.7, 0.8, 1.0); // Light blue
    let day_zenith = vec3(0.2, 0.4, 0.8); // Deep blue
    let sunset_horizon = vec3(1.0, 0.5, 0.0); // Orange
    let sunset_zenith = vec3(0.2, 0.0, 0.4); // Purple
    let night_horizon = vec3(0.0, 0.0, 0.1); // Dark blue
    let night_zenith = vec3(0.0, 0.0, 0.0); // Black

    // Smooth transitions
    let day_to_sunset = smoothstep(-0.1, 0.1, sun_dot); // Day (1) to sunset (0)
    let sunset_to_night = smoothstep(-0.3, -0.1, sun_dot); // Sunset (1) to night (0)

    // Blend horizon and zenith colors
    let horizon_color = mix(
        mix(night_horizon, sunset_horizon, sunset_to_night),
        day_horizon,
        day_to_sunset
    );
    let zenith_color = mix(
        mix(night_zenith, sunset_zenith, sunset_to_night),
        day_zenith,
        day_to_sunset
    );

    // Rayleigh-like phase for sky scattering
    let rayleigh_phase = 0.75 * (1.0 + view_sun_dot2); // Simplified from Shadertoy

    // Base sky gradient
    let sky_col = mix(horizon_color, zenith_color, t) * rayleigh_phase;

    // Sun-side orange during sunset
    let sunset_factor = 1.0 - day_to_sunset; // Stronger during sunset
    let sun_proximity = smoothstep(0.8, 1.0, view_sun_dot); // Near sun
    let sunset_glow = vec3(1.0, 0.6, 0.2) * sun_proximity * sunset_factor * 0.5 * globals.sun_intensity;
    let sky_with_glow = sky_col + sunset_glow;

    return clamp(sky_with_glow, vec3(0.0), vec3(1.0));
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
    let ro = view.world_position;
    let rd_vec = world_pos3 - ro;
    let t_max = length(rd_vec);
    let rd = rd_vec / t_max;

    let dither = fract(blue_noise(pix));

    var accumulation = vec4(0.0);
    var t = dither * STEP_SIZE_INSIDE;
    var steps_outside_cloud = 0;

    // Get sun direction and intensity, mix between aur light (straight up) and sun
    let sun_dot = globals.sun_direction.y;
    let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    let sun_dir = normalize(mix(AUR_DIR, globals.sun_direction, sun_t));

    // Compute scattering angle (dot product between view direction and light direction)
    let scattering_angle = dot(rd, sun_dir);

    // Calculate cylinders to render for the poles, behind all clouds
    let pole_color = render_poles(ro, rd, view.planet_rotation, view.world_position, globals.planet_radius);

    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= t_max || accumulation.a >= ALPHA_THRESHOLD || t >= FOG_END_DISTANCE {
            break;
        }

        let pos = ro + rd * t;
        let cloud_sample = sample_cloud(pos, t);
        let step_density = cloud_sample.density;

        // Scale step size based on distance from camera
        var step_scaler = 1.0;
        if t > STEP_DISTANCE_SCALING_START {
            step_scaler = 1.0 + (t - STEP_DISTANCE_SCALING_START) * STEP_DISTANCE_SCALING_FACTOR;
        }
        // Reduce scaling when close to surfaces
        let close_threshold = STEP_SIZE_OUTSIDE * step_scaler;
        let distance_left = t_max - t;
        if distance_left < close_threshold {
            let norm = clamp(distance_left / close_threshold, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }

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

        var step = STEP_SIZE_OUTSIDE * step_scaler;
        if step_density > 0.0 {
            step = STEP_SIZE_INSIDE * step_scaler;

            let step_color = cloud_sample.color;

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var light_step_size = LIGHT_STEP_SIZE;
            for (var j: i32 = 1; j <= LIGHT_STEPS; j += 1) {
                let light_offset = pos + sun_dir * f32(j) * light_step_size;
                density_sunwards += sample_cloud(light_offset, t).density * light_step_size;
                if density_sunwards >= 0.95 {
                    break;
                }
            }

            // Calcuate self shadowing
            let tau = clamp(density_sunwards, 0.0, 1.0) * SHADOW_EXTINCTION;

            // Compute sky color for the current view direction
            let sky_col = sky_color(rd, globals.sun_direction);

            // Height factor for reducing aur light
            let height_factor = max(smoothstep(15000.0, 100.0, pos.y), 0.15);

            // Blend between aur and sun, modulated by sky color
            let sun_color = mix(AUR_COLOR * height_factor, SUN_COLOR * globals.sun_intensity, sun_t);
            let sky_sun_blend = mix(sky_col, vec3(1.0), sun_t); // Full sky color at sunset, white at day
            let sun_color_with_sky = sun_color * sky_sun_blend;

            // Blend between sky based ambient and aur based on sun height, always using a little aur light
            let ambient_color = (AMBIENT_AUR_COLOR * height_factor * 20.0) + mix(AMBIENT_AUR_COLOR * height_factor, sky_col, max(sun_t - 0.2, 0.0));

            // Apply transmission to sun and ambient light
            let transmitted_sun = transmission(sun_color, tau);
            let transmitted_ambient = transmission(ambient_color, tau * 0.5); // Ambient attenuated less

            // Apply light scattering to sun light based on angle
            let scattered_sun = light_scattering(sun_color, scattering_angle);

            // Combine transmitted and scattered light, weighted by density
            var lit_color = ((transmitted_sun + scattered_sun) * step_density + transmitted_ambient) * step_color + cloud_sample.emission;

            let step_alpha = clamp(step_density * 0.4 * step, 0.0, 1.0);
            accumulation += vec4(lit_color * step_alpha * (1.0 - accumulation.a), step_alpha * (1.0 - accumulation.a));
        }

        t += step;
    }

    accumulation.a = min(accumulation.a * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    if depth <= 0.00001 {
        // Blend in poles behind clouds
        accumulation = vec4(
            accumulation.rgb + pole_color.rgb * pole_color.a * (1.0 - accumulation.a),
            accumulation.a + pole_color.a * (1.0 - accumulation.a)
        );

        // Add a small amount of our sky color over the bevy sky
        // let sky_col = sky_color(rd, globals.sun_direction);
        // accumulation = vec4(
        //     accumulation.rgb + sky_col * 0.8 * (1.0 - accumulation.a),
        //     accumulation.a + 0.8 * (1.0 - accumulation.a)
        // );
    }

    return clamp(accumulation, vec4(0.0), vec4(1.0));
}
