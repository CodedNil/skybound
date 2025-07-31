#define_import_path skybound::functions

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
