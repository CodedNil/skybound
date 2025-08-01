#define_import_path skybound::functions

// Many hash functions https://www.shadertoy.com/view/XlGcRh

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

// White noise hash: vec3 → vec3 [-1,1]
const UI3: vec3<u32> = vec3<u32>(1597334673u, 3812015801u, 2798796415u);
const UIF: f32 = 1.0 / f32(0xffffffffu);
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

// Perlin tiled noise: vec3 f32 → f32 [-1,1]
fn perlin31(x: vec3<f32>) -> f32 {
    let p: vec3<f32> = floor(x);
    let w: vec3<f32> = fract(x);

    // Quintic interpolant
    let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    // Gradients
    let ga: vec3<f32> = hash33(p + vec3<f32>(0.0, 0.0, 0.0));
    let gb: vec3<f32> = hash33(p + vec3<f32>(1.0, 0.0, 0.0));
    let gc: vec3<f32> = hash33(p + vec3<f32>(0.0, 1.0, 0.0));
    let gd: vec3<f32> = hash33(p + vec3<f32>(1.0, 1.0, 0.0));
    let ge: vec3<f32> = hash33(p + vec3<f32>(0.0, 0.0, 1.0));
    let gf: vec3<f32> = hash33(p + vec3<f32>(1.0, 0.0, 1.0));
    let gg: vec3<f32> = hash33(p + vec3<f32>(0.0, 1.0, 1.0));
    let gh: vec3<f32> = hash33(p + vec3<f32>(1.0, 1.0, 1.0));

    // Projections
    let va: f32 = dot(ga, w - vec3<f32>(0.0, 0.0, 0.0));
    let vb: f32 = dot(gb, w - vec3<f32>(1.0, 0.0, 0.0));
    let vc: f32 = dot(gc, w - vec3<f32>(0.0, 1.0, 0.0));
    let vd: f32 = dot(gd, w - vec3<f32>(1.0, 1.0, 0.0));
    let ve: f32 = dot(ge, w - vec3<f32>(0.0, 0.0, 1.0));
    let vf: f32 = dot(gf, w - vec3<f32>(1.0, 0.0, 1.0));
    let vg: f32 = dot(gg, w - vec3<f32>(0.0, 1.0, 1.0));
    let vh: f32 = dot(gh, w - vec3<f32>(1.0, 1.0, 1.0));

    // Interpolation
    return va +
           u.x * (vb - va) +
           u.y * (vc - va) +
           u.z * (ve - va) +
           u.x * u.y * (va - vb - vc + vd) +
           u.y * u.z * (va - vc - ve + vg) +
           u.z * u.x * (va - vb - ve + vf) +
           u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Cellular noise: vec3<f32>, f32 → f32 [0,1]
fn worley31(p: vec3<f32>) -> f32 {
    let id: vec3<i32> = vec3<i32>(floor(p));
    let frac: vec3<f32> = fract(p);
    var md: f32 = 1e4;

    for (var x: i32 = -1; x <= 1; x++) {
        for (var y: i32 = -1; y <= 1; y++) {
            for (var z: i32 = -1; z <= 1; z++) {
                let off: vec3<i32> = vec3<i32>(x,y,z);
                let rnd: vec3<f32> = hash33(vec3<f32>(id + off)) * 0.5 + 0.5;
                let d: vec3<f32> = frac - (vec3<f32>(off) + rnd);
                md = min(md, dot(d, d));
            }
        }
    }
    return 1.0 - md;
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

// Value-FBM combined: vec3<f32> u32 → f32 [-1,1]
fn value_fbm31(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += WEIGHTS[i] * value31(pos);
        pos = pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Perlin-FBM combined: vec3<f32> u32 → f32 [-1,1]
const AMPLITUDE_DECAY: f32 = exp2(-0.85);
fn perlin_fbm31(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += WEIGHTS[i] * perlin31(pos);
        pos = pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Worley-FBM combined: vec3<f32> u32 → f32 [-1,1]
fn worley_fbm31(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += WEIGHTS[i] * worley31(pos);
        pos = pos * M3;
    }
    return sum / NORMS[octaves - 1u];
}
