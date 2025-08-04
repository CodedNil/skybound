#define_import_path skybound::functions

// Many hash functions https://www.shadertoy.com/view/XlGcRh
// https://github.com/johanhelsing/noisy_bevy/blob/main/assets/noisy_bevy.wgsl

// Remap a range
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return (((x - a) / (b - a)) * (d - c)) + c;
}

// Modulo function: vec3, f32 → vec3
fn mod3(x: vec3<f32>, y: f32) -> vec3<f32> {
    return x - floor(x / y) * y;
}

// White noise hash: f32 → f32 [0,1]
fn hash11(p: f32) -> f32 {
    var v: f32 = fract(p * 0.1031);
    v *= v + 33.33;
    v *= v + v;
    return fract(v);
}

// White noise hash: f32 → vec2 [0,1]
fn hash12(p: f32) -> vec2<f32> {
    var v: vec2<f32> = fract(vec2<f32>(p) * vec2<f32>(0.1031, 0.1030));
    v += dot(v, v.yx + 33.33);
    return fract((v.x + v.y) * v);
}

// White noise hash: vec2 → f32 [0,1]
fn hash21(p: vec2<f32>) -> f32 {
    var v3: vec3<f32> = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    v3 += dot(v3, v3.yzx + 33.33);
    return fract((v3.x + v3.y) * v3.z);
}

// White noise integer hash: vec3i → f32 [-1,1]
const HASH_MULTIPLIER: f32 = 1.0 / f32(0x0fffffff);
fn hash3i1(p: vec3<i32>) -> f32 {
    var n: i32 = p.x * 3 + p.y * 113 + p.z * 311;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * f32(n & 0x0fffffff) * HASH_MULTIPLIER;
}

// White noise integer hash: vec2 → vec2 [-1,1]
const UI2_1: vec2<u32> = vec2<u32>(1597334673u, 3812015801u);
const UI2_2: vec2<u32> = vec2<u32>(362436069u, 521288629u);
const UIF: f32 = 1.0 / f32(0xffffffffu);
fn hash22(p: vec2<f32>) -> vec2<f32> {
    let pi_i: vec2<i32> = vec2<i32>(floor(p));
    var pi_u: vec2<u32> = vec2<u32>(pi_i) * UI2_1;
    let h: u32 = pi_u.x ^ pi_u.y;
    var qi_u: vec2<u32>  = vec2<u32>(h) * UI2_2;
    return (vec2<f32>(qi_u) * UIF) * 2.0 - 1.0;
}

// White noise integer hash: vec3 → vec3 [-1,1]
const UI3: vec3<u32> = vec3<u32>(1597334673u, 3812015801u, 2798796415u);
fn hash33(p: vec3<f32>) -> vec3<f32> {
    var qi: vec3<u32> = vec3<u32>(vec3<i32>(p)) * UI3;
    var q2: vec3<u32> = vec3<u32>(qi.x ^ qi.y ^ qi.z) * UI3;
    return vec3<f32>(q2) * UIF * 2.0 - 1.0;
}

// Blue noise approx.: vec2 → f32 [0,1]
fn blue_noise(uv: vec2<f32>) -> f32 {
    var s0: f32 = hash21(uv + vec2<f32>(-1.0, 0.0));
    var s1: f32 = hash21(uv + vec2<f32>(1.0, 0.0));
    var s2: f32 = hash21(uv + vec2<f32>(0.0, 1.0));
    var s3: f32 = hash21(uv + vec2<f32>( 0.0, -1.0));
    var s: f32 = s0 + s1 + s2 + s3;
    return hash21(uv) - s * 0.25 + 0.5;
}

// Value noise: vec3 → f32 [-1,1]
fn value31(p: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(p));
    let f: vec3<f32> = fract(p);
    let u: vec3<f32> = f * f * (vec3<f32>(3.0) - 2.0 * f);

    return mix(
        // Mix along y-axis for the bottom face (z=0)
        mix(
            mix(hash3i1(i + vec3<i32>(0,0,0)), hash3i1(i + vec3<i32>(1,0,0)), u.x),
            mix(hash3i1(i + vec3<i32>(0,1,0)), hash3i1(i + vec3<i32>(1,1,0)), u.x),
            u.y
        ),
        // Mix along y-axis for the top face (z=1)
        mix(
            mix(hash3i1(i + vec3<i32>(0,0,1)), hash3i1(i + vec3<i32>(1,0,1)), u.x),
            mix(hash3i1(i + vec3<i32>(0,1,1)), hash3i1(i + vec3<i32>(1,1,1)), u.x),
            u.y
        ),
        u.z // Mix along z-axis between the two faces
    );
}

// Perlin noise: vec3 → f32 [-1,1]
fn perlin31(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Quintic interpolant
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let g000 = hash33(i);
    let g100 = hash33(i + vec3<f32>(1.0, 0.0, 0.0));
    let g010 = hash33(i + vec3<f32>(0.0, 1.0, 0.0));
    let g110 = hash33(i + vec3<f32>(1.0, 1.0, 0.0));
    let g001 = hash33(i + vec3<f32>(0.0, 0.0, 1.0));
    let g101 = hash33(i + vec3<f32>(1.0, 0.0, 1.0));
    let g011 = hash33(i + vec3<f32>(0.0, 1.0, 1.0));
    let g111 = hash33(i + vec3<f32>(1.0, 1.0, 1.0));

    let n000 = dot(g000, f);
    let n100 = dot(g100, f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = dot(g010, f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = dot(g110, f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = dot(g001, f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = dot(g101, f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = dot(g011, f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = dot(g111, f - vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}

// Cellular noise: vec2 → f32 [0,1]
fn worley21(p: vec2<f32>) -> f32 {
    let cell: vec2<f32> = floor(p);
    let fractal: vec2<f32> = p - cell;
    var min_dist: f32 = 1e10;

    // Search the 3×3 neighborhood
    for (var x = -1.0; x <= 1.0; x += 1.0) {
        for (var y = -1.0; y <= 1.0; y += 1.0) {
            let offset: vec2<f32> = vec2<f32>(x, y);
            let point: vec2<f32> = hash22(cell + offset) * 0.5 + 0.5 + offset;
            let dist: f32 = length(fractal - point);
            min_dist = min(min_dist, dist);
        }
    }

    return 1.0 - min_dist;
}

// Cellular noise: vec3 → f32 [0,1]
fn worley31(p: vec3<f32>) -> f32 {
    let cell: vec3<f32> = floor(p);
    let fractal: vec3<f32> = p - cell;
    var min_dist: f32 = 1e4;

    for (var x = -1.0; x <= 1.0; x += 1.0) {
        for (var y = -1.0; y <= 1.0; y += 1.0) {
            for (var z = -1.0; z <= 1.0; z += 1.0) {
                let offset: vec3<f32> = vec3<f32>(x, y, z);
                let point: vec3<f32> = hash33(cell + offset) * 0.5 + 0.5 + offset;
                let dist = fractal - point;
                min_dist = min(min_dist, dot(dist, dist));
            }
        }
    }

    return 1.0 - min_dist;
}

// Fractional Brownian motion: vec3 → f32 [-1,1]
const M3: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.8, 0.6, 0.0),
    vec3<f32>(-0.6, 0.8, 0.0),
    vec3<f32>(0.0, 0.0, 1.0)
) * 2.0;
const MAX_OCT: u32 = 6u;
const WEIGHTS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625);
const NORMS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(1.0, 0.75, 0.875, 0.9375, 0.96875, 0.984375);

// Value-FBM combined: vec3 u32 → f32 [-1,1]
fn value_fbm31(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += WEIGHTS[i] * value31(pos);
        pos = pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Perlin-FBM combined: vec3 u32 → f32 [-1,1]
fn perlin_fbm31(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += WEIGHTS[i] * perlin31(pos);
        pos = pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Worley-FBM combined: vec3 → f32 [0,1]
fn worley_fbm31(p: vec3<f32>) -> f32 {
    return worley31(p) * 0.625 + worley31(p * 2.0) * 0.25 + worley31(p * 4.0) * 0.125;
}
