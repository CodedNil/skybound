#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 3.0;
const STEP_SIZE_OUTSIDE: f32 = 10.0;

const STEP_DISTANCE_SCALING_START: f32 = 100.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.001; // How much to scale step size by distance

const LIGHT_STEPS: i32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 3.0;

// Lighting variables
const LIGHT_DIRECTION: vec3<f32> = vec3(0.0, 0.89, 0.45);
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);
const AMBIENT_COLOR: vec3<f32> = vec3(0.7, 0.8, 1.0) * 0.25;
const FOG_START_DISTANCE: f32 = 1000.0;
const FOG_END_DISTANCE: f32 = 5000.0;
const SHADOW_EXTINCTION: f32 = 5.0; // Higher = deeper core shadows

// Cloud Material Parameters
const BACK_SCATTERING: f32 = 1.0; // Backscattering
const BACK_SCATTERING_FALLOFF: f32 = 30.0; // Backscattering falloff
const OMNI_SCATTERING: f32 = 0.8; // Omnidirectional Scattering
const TRANSMISSION_SCATTERING: f32 = 1.0; // Transmission Scattering
const TRANSMISSION_FALLOFF: f32 = 2.0; // Transmission falloff
const BASE_TRANSMISSION: f32 = 0.1; // Light that doesn't get scattered at all

// Turbulence parameters for fog
const FOG_COLOR_A: vec3<f32> = vec3(0.3, 0.2, 0.8); // Deep blue
const FOG_COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6); // Deep purple
const FOG_ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const FOG_TURB_ITERS: f32 = 8.0; // Number of turbulence iterations
const FOG_TURB_AMP: f32 = 0.6; // Turbulence amplitude
const FOG_TURB_SPEED: f32 = 0.5; // Turbulence speed
const FOG_TURB_FREQ: f32 = 0.4; // Initial turbulence frequency
const FOG_TURB_EXP: f32 = 1.6; // Frequency multiplier per iteration

// Fog lightning parameters
const FOG_FLASH_FREQUENCY: f32 = 0.2; // Chance of a flash per second per cell
const FOG_FLASH_GRID: f32 = 2000.0; // Grid cell size
const FOG_FLASH_COLOR: vec3<f32> = vec3(0.8, 0.9, 1.0);
const FOG_FLASH_DURATION: f32 = 6.0; // Seconds
const FOG_FLASH_FLICKER_SPEED: f32 = 12.0; // Hz of the on/off cycles

// Simple noise functions for white noise
fn hash_12(p: f32) -> vec2<f32> {
    var p2 = fract(vec2(p) * vec2(0.1031, 0.1030));
    p2 += dot(p2, p2.yx + 33.33);
    return fract((p2.x + p2.y) * p2);
}

fn hash_21(pos: vec2<f32>) -> f32 {
    var p_3 = fract(vec3(pos.x, pos.y, pos.x) * 0.1031);
    p_3 += dot(p_3, p_3.yzx + 33.33);
    return fract((p_3.x + p_3.y) * p_3.z);
}

fn hash_3i1(pos: vec3<i32>) -> f32 {
    var n: i32 = pos.x * 3 + pos.y * 113 + pos.z * 311;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * f32(n & 0x0fffffff) / f32(0x0fffffff);
}

// Procedural blue noise approximation
fn blue_noise(uv: vec2<f32>) -> f32 {
    let v = hash_21(uv + vec2(-1.0, 0.0)) + hash_21(uv + vec2(1.0, 0.0)) + hash_21(uv + vec2(0.0, 1.0)) + hash_21(uv + vec2(0.0, -1.0));
    return hash_21(uv) - v * 0.25 + 0.5;
}

// Simple noise functions
fn noise_3(pos: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(pos));
    let f: vec3<f32> = fract(pos);
    let u: vec3<f32> = f * f * (3.0 - 2.0 * f); // Smoothstep weights

    // Hash values at cube corners, interpolate along x
    let lerp_x_0 = mix(hash_3i1(i + vec3<i32>(0, 0, 0)), hash_3i1(i + vec3<i32>(1, 0, 0)), u.x);
    let lerp_x_1 = mix(hash_3i1(i + vec3<i32>(0, 1, 0)), hash_3i1(i + vec3<i32>(1, 1, 0)), u.x);
    let lerp_x_2 = mix(hash_3i1(i + vec3<i32>(0, 0, 1)), hash_3i1(i + vec3<i32>(1, 0, 1)), u.x);
    let lerp_x_3 = mix(hash_3i1(i + vec3<i32>(0, 1, 1)), hash_3i1(i + vec3<i32>(1, 1, 1)), u.x);

    // Interpolate along y
    let lerp_y_0 = mix(lerp_x_0, lerp_x_1, u.y);
    let lerp_y_1 = mix(lerp_x_2, lerp_x_3, u.y);

    // Interpolate along z and return
    return mix(lerp_y_0, lerp_y_1, u.z);
}

// Fractional Brownian Motion (FBM)
const M3: mat3x3<f32> = mat3x3<f32>(
    vec3(0.8, 0.6, 0.0),
    vec3(-0.6, 0.8, 0.0),
    vec3(0.0, 0.0, 1.0)
) * 2.0;
const MAX_OCT: u32 = 6u;
const WEIGHTS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625);
const NORMS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(1.0, 0.75, 0.875, 0.9375, 0.96875, 0.984375);
fn fbm_3(pos: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var freq_pos = pos;
    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        sum += WEIGHTS[i] * noise_3(freq_pos);
        freq_pos = freq_pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Sky shading
fn render_sky(rd: vec3<f32>, sun_direction: f32) -> vec3<f32> {
    let elevation = 1.0 - dot(rd, vec3(0.0, 1.0, 0.0));
    let centered = 1.0 - abs(1.0 - elevation);

    let atmosphere_color = mix(AMBIENT_COLOR, SUN_COLOR, sun_direction * 0.5);
    let base = mix(pow(AMBIENT_COLOR, vec3(4.0)), atmosphere_color, pow(clamp(elevation, 0.0, 1.0), 0.5));
    let haze = pow(centered + 0.02, 4.0) * (sun_direction * 0.2 + 0.8);

    let sky = mix(base, SUN_COLOR, clamp(haze, 0.0, 1.0));
    let sun = pow(max((sun_direction - 29.0 / 30.0) * 30.0 - 0.05, 0.0), 6.0);

    return sky + vec3(sun);
}

// Fog turbulence and lightning calculations
fn fog_compute_turbulence(initial_pos: vec2<f32>) -> vec2<f32> {
    var pos = initial_pos;
    var freq = FOG_TURB_FREQ;
    var rot = FOG_ROTATION_MATRIX;
    for (var i = 0.0; i < FOG_TURB_ITERS; i = i + 1.0) {
        // Compute phase using rotated y-coordinate, time, and iteration offset
        let phase = freq * (pos * rot).y + FOG_TURB_SPEED * globals.time + i;
        pos = pos + FOG_TURB_AMP * rot[0] * sin(phase) / freq; // Add perpendicular sine offset
        rot = rot * FOG_ROTATION_MATRIX; // Rotate for next iteration
        freq = freq * FOG_TURB_EXP; // Increase frequency
    }

    return pos;
}

// Voronoi-style closest point calculation for fog
fn fog_flash_emission(pos: vec3<f32>) -> vec3<f32> {
    let cell = floor(pos.xz / FOG_FLASH_GRID);
    var min_dist = 1e10;
    var is_flashing = false;

    // Find closest flashing point across 3x3 grid
    for (var i = -1.0; i <= 1.0; i = i + 1.0) {
        for (var j = -1.0; j <= 1.0; j = j + 1.0) {
            let neighbor = cell + vec2(i, j);
            let seed = dot(neighbor, vec2(127.1, 311.7));
            let flash_seed = seed + floor(globals.time / FOG_FLASH_DURATION);
            let within_duration = (globals.time - floor(globals.time / FOG_FLASH_DURATION) * FOG_FLASH_DURATION) < FOG_FLASH_DURATION;
            let flicker = hash_12(flash_seed + floor(globals.time * FOG_FLASH_FLICKER_SPEED)).x > 0.5;
            if within_duration && flicker && hash_12(flash_seed).x <= FOG_FLASH_FREQUENCY {
                let offset = hash_12(seed) * 0.5 + 0.5 * sin(globals.time * 0.1 + 6.2831 * hash_12(seed));
                let point = vec3((neighbor + offset) * FOG_FLASH_GRID, -125.0);
                let dist = length(pos - point);
                if dist < min_dist {
                    min_dist = dist;
                    is_flashing = true;
                }
            }
        }
    }

    if !is_flashing {
        return vec3(0.0);
    }

    return exp(-min_dist / 50.0) * FOG_FLASH_COLOR * 10.0;
}


// Cloud density and colour
struct CloudSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_cloud(pos: vec3<f32>) -> CloudSample {
    var sample: CloudSample;
    let altitude = pos.y;

    // Thick aur fog below 0m
    let fog_density = smoothstep(20.0, -100.0, altitude);
    var fog_contribution = 0.0;
    var fog_color = vec3(1.0);
    if fog_density > 0.01 && pos.y > -250.0 {
        // Use turbulent position for density
        let turb_pos = fog_compute_turbulence(pos.xz * 0.05);
        let fbm_value = fbm_3(vec3(turb_pos.x, pos.y * 0.05, turb_pos.y), 4);
        fog_contribution = pow(fbm_value, 2.0) * fog_density + smoothstep(-50.0, -200.0, altitude);

        // Compute fog color based on turbulent flow
        fog_color = mix(FOG_COLOR_A, FOG_COLOR_B, fbm_value);

        // Apply artificial shadowing: darken towards black as altitude decreases
        let shadow_factor = 1.0 - smoothstep(0.0, -50.0, altitude);
        fog_color = mix(fog_color * 0.1, fog_color, shadow_factor);

        // Compute lightning emission using Voronoi grid
        sample.emission = fog_flash_emission(pos) * fog_contribution;
    }

    // Low clouds starting at y=200
    let low_gradient = smoothstep(200.0, 300.0, altitude) * smoothstep(800.0, 700.0, altitude);
    var cloud_contribution = 0.0;

    if low_gradient > 0.01 {
        let low_type = smoothstep(200.0, 500.0, altitude) * smoothstep(3000.0, 2000.0, altitude) * (1.0 + 0.5 * sin(altitude * 0.002));
        let base_noise = fbm_3(pos * 0.01 + vec3(0.0, globals.time * 0.02, 0.0), 6);
        let shaped_noise = base_noise * low_type;
        cloud_contribution = shaped_noise * low_gradient;
    }

    let total_density = fog_contribution + cloud_contribution;
    sample.density = total_density;

    if total_density > 0.0 {
        let cloud_color = vec3(1.0); // White for clouds
        sample.color = (fog_color * fog_contribution + cloud_color * cloud_contribution) / total_density;
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

// Raymarch through all the clouds
fn raymarch(uv: vec2<f32>, pix: vec2<f32>) -> vec4<f32> {
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

    let sun_direction = dot(rd, LIGHT_DIRECTION);
    let sky = render_sky(rd, sun_direction);

    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= t_max || accumulation.a >= ALPHA_THRESHOLD {
            break;
        }

        let pos = ro + rd * t;
        let cloud_sample = sample_cloud(pos);
        let step_density = cloud_sample.density;

        // Scale step size based on distance from camera
        var step_scaler = 1.0;
        if t > STEP_DISTANCE_SCALING_START {
            step_scaler = 1.0 + (t - STEP_DISTANCE_SCALING_START) * STEP_DISTANCE_SCALING_FACTOR;
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
                let light_offset = pos + LIGHT_DIRECTION * f32(j) * light_step_size;
                density_sunwards += sample_cloud(light_offset).density * light_step_size;
                if density_sunwards >= 0.95 {
                    break;
                }
            }

            let tau = clamp(density_sunwards, 0.0, 1.0);
            let self_shadow = exp(-SHADOW_EXTINCTION * tau); // Inner shadow darkening, with Beer function

            // Final color with self-shadowing
            let lit_color = mix(AMBIENT_COLOR, SUN_COLOR, self_shadow) * step_color + cloud_sample.emission;

            let step_alpha = clamp(step_density * 0.4 * step, 0.0, 1.0);
            accumulation += vec4(lit_color * step_alpha * (1.0 - accumulation.a), step_alpha * (1.0 - accumulation.a));
        }

        t += step;
    }

    accumulation.a = min(accumulation.a * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    // Add sky, taking into account t_max for distance fog effect
    // let sky_alpha_factor = smoothstep(FOG_START_DISTANCE, FOG_END_DISTANCE, t_max);
    // accumulation += vec4(beer(accumulation.a * (1.0 - BASE_TRANSMISSION)) * sky, sky_alpha_factor);

    return clamp(accumulation, vec4(0.0), vec4(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;

    return raymarch(uv, pix);
}
