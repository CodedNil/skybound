#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_2d;
@group(0) @binding(4) var history_texture: texture_2d<f32>;
@group(0) @binding(5) var<uniform> previous_view_projection: mat4x4<f32>;

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.9; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 128;
const STEP_SIZE_INSIDE: f32 = 2.0;
const STEP_SIZE_OUTSIDE: f32 = 8.0;

const STEP_DISTANCE_SCALING_START: f32 = 100.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.005; // How much to scale step size by distance

const LIGHT_STEPS: i32 = 2; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 16.0;

// Lighting variables
const LIGHT_DIRECTION: vec3<f32> = vec3<f32>(0.0, 0.89, 0.45);
const SUN_COLOR: vec3<f32> = vec3<f32>(0.99, 0.97, 0.96);
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.52, 0.80, 0.92);
const FOG_START_DISTANCE: f32 = 1000.0;
const FOG_END_DISTANCE: f32 = 5000.0;

// Cloud Material Parameters
const BACK_SCATTERING: f32 = 1.0; // Backscattering
const BACK_SCATTERING_FALLOFF: f32 = 30.0; // Backscattering falloff
const OMNI_SCATTERING: f32 = 0.8; // Omnidirectional Scattering
const TRANSMISSION_SCATTERING: f32 = 1.0; // Transmission Scattering
const TRANSMISSION_FALLOFF: f32 = 2.0; // Transmission falloff
const BASE_TRANSMISSION: f32 = 0.1; // Light that doesn't get scattered at all

// Bayer matrix for 4x4 pattern
const BAYER_LIMIT: u32 = 16;
const BAYER_LIMIT_H: u32 = 4;
const BAYER_FILTER: array<u32, BAYER_LIMIT> = array<u32, BAYER_LIMIT>(
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5
);

// Simple noise function for white noise
fn hash_2(pos: vec2<f32>) -> f32 {
    var p_3 = fract(vec3(pos.x, pos.y, pos.x) * 0.1031);
    p_3 += dot(p_3, p_3.yzx + 33.33);
    return fract((p_3.x + p_3.y) * p_3.z);
}

fn hash_3i(pos: vec3<i32>) -> f32 {
    var n: i32 = pos.x * 3 + pos.y * 113 + pos.z * 311;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * f32(n & 0x0fffffff) / f32(0x0fffffff);
}

fn hash_4(pos: vec4<f32>) -> f32 {
    var p_4 = fract(pos * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p_4 += dot(p_4, p_4.wzxy + 33.33);
    return fract((p_4.x + p_4.y) * (p_4.z + p_4.w));
}

// Procedural blue noise approximation
fn blue_noise(uv: vec2<f32>) -> f32 {
    let v = hash_2(uv + vec2(-1.0, 0.0)) + hash_2(uv + vec2(1.0, 0.0)) + hash_2(uv + vec2(0.0, 1.0)) + hash_2(uv + vec2(0.0, -1.0));
    return hash_2(uv) - v * 0.25 + 0.5;
}

// Simple noise functions
fn noise_3(pos: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(pos));
    let f: vec3<f32> = fract(pos);
    let u: vec3<f32> = f * f * (3.0 - 2.0 * f); // Smoothstep weights

    // Hash values at cube corners, interpolate along x
    let lerp_x_0 = mix(hash_3i(i + vec3<i32>(0, 0, 0)), hash_3i(i + vec3<i32>(1, 0, 0)), u.x);
    let lerp_x_1 = mix(hash_3i(i + vec3<i32>(0, 1, 0)), hash_3i(i + vec3<i32>(1, 1, 0)), u.x);
    let lerp_x_2 = mix(hash_3i(i + vec3<i32>(0, 0, 1)), hash_3i(i + vec3<i32>(1, 0, 1)), u.x);
    let lerp_x_3 = mix(hash_3i(i + vec3<i32>(0, 1, 1)), hash_3i(i + vec3<i32>(1, 1, 1)), u.x);

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

const MAX_OCT: u32 = 5u;
const WEIGHTS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(0.5, 0.25, 0.125, 0.0625, 0.03125);
const NORMS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(1.0, 0.75, 0.875, 0.9375, 0.96875);

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

// Cloud density
fn density_at_cloud(pos: vec3<f32>) -> f32 {
    let clouds_type: f32 = 0.0;
    let clouds_coverage: f32 = 1.0;

    let altitude = pos.y;
    var density = 0.0;

    // Thick aur fog below 0m
    let fog_density = smoothstep(0.0, -10.0, pos.y);
    if fog_density > 0.01 {
        let fbm_value = fbm_3(pos / 5.0 - vec3(0.0, 0.1, 1.0) * globals.time, 3);
        density = fbm_value * fog_density + fog_density * 0.5;
    }

    let low_gradient = smoothstep(60.0, 200.0, pos.y) * smoothstep(800.0, 700.0, pos.y);
    let mid_gradient = smoothstep(700.0, 800.0, pos.y) * smoothstep(1800.0, 1700.0, pos.y);
    let high_gradient = smoothstep(1700.0, 1800.0, pos.y) * smoothstep(2700.0, 2500.0, pos.y);

    // Cloud type blending
    let low_type = mix(
        // Stratus: Flat, layered profile
        smoothstep(0.0, 200.0, altitude) * smoothstep(1500.0, 1000.0, altitude),
        // Cumulus: Puffy, vertical development
        smoothstep(200.0, 500.0, altitude) * smoothstep(3000.0, 2000.0, altitude) * (1.0 + 0.5 * sin(altitude * 0.002)),
        saturate(clouds_type * 2.0)
    );
    // Cirrus: Thin, wispy profile
    let high_type = smoothstep(5000.0, 6000.0, altitude) * smoothstep(12000.0, 10000.0, altitude) * 0.3;

    // Generate base cloud shapes
    if low_gradient > 0.01 {
        let base_noise = fbm_3(pos * 0.005 + vec3(0.0, globals.time * 0.02, 0.0), 5);
        let shaped_noise = base_noise * low_type;
        density += shaped_noise * low_gradient * clouds_coverage;
    }
    // if mid_gradient > 0.01 {
    //     let base_noise = fbm_3(pos * 0.004 + vec3(0.0, globals.time * 0.02, 0.0), 4);
    //     let shaped_noise = base_noise * mix(low_type, high_type, 0.5);
    //     density += shaped_noise * mid_gradient * clouds_coverage;
    // }
    // if high_gradient > 0.01 {
    //     let base_noise = fbm_3(pos * 0.003 + vec3(0.0, globals.time * 0.02, 0.0), 4);
    //     let shaped_noise = base_noise * high_type;
    //     density += shaped_noise * high_gradient * clouds_coverage * 0.7; // Thinner high clouds
    // }

    return clamp(density, 0.0, 1.0);
}

// Lighting Functions
fn compute_density_towards_sun(pos: vec3<f32>, density_here: f32) -> f32 {
    var density_sunwards = max(density_here, 0.0);
    for (var j: i32 = 1; j <= LIGHT_STEPS; j = j + 1) {
        let light_offset = pos + LIGHT_DIRECTION * f32(j) * LIGHT_STEP_SIZE;
        density_sunwards += density_at_cloud(light_offset) * LIGHT_STEP_SIZE;
    }

    return density_sunwards;
}

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
        let step_density = density_at_cloud(pos);

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

            let material_here = step_density * step;
            let material_towards_sun = compute_density_towards_sun(pos, step_density);
            let light_at_particle = transmission(SUN_COLOR, material_towards_sun);

            let light_scattering_towards_camera = light_scattering(light_at_particle * material_here, sun_direction);
            let light_reaching_camera = transmission(light_scattering_towards_camera, accumulation.a + material_here);
            accumulation += vec4(light_reaching_camera, material_here);
        }

        t += step;
    }

    accumulation.a = min(accumulation.a * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    // Add sky, taking into account t_max for distance fog effect
    let sky_alpha_factor = smoothstep(FOG_START_DISTANCE, FOG_END_DISTANCE, t_max);
    accumulation += vec4(beer(accumulation.a * (1.0 - BASE_TRANSMISSION)) * sky, sky_alpha_factor); // Add sky

    return clamp(accumulation, vec4(0.0), vec4(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;

    // Bayer matrix to select 1/16th of pixels
    let bayer_index = (u32(pix.x) % BAYER_LIMIT_H) * BAYER_LIMIT_H + (u32(pix.y) % BAYER_LIMIT_H);
    let should_render = (globals.frame_count % BAYER_LIMIT) == BAYER_FILTER[bayer_index];

    // Render new pixels or use history
    if globals.frame_count == 0u || should_render {
        return raymarch(uv, pix); // Raymarch for 1/16th of pixels
    }

    // Reconstruct world‑space pos
    let ndc = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 1.0, 1.0);
    let world_pos4 = view.world_from_clip * ndc;

    // Reproject to previous UV
    let prev_clip = previous_view_projection * world_pos4;
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));

    if prev_clip.w <= 0.0 || prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        return raymarch(uv, pix); // Fall back to raymarching
    }

    // Sample history
    return textureSample(history_texture, linear_sampler, prev_uv);
}
