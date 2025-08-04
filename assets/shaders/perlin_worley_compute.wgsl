@group(0) @binding(0) var output: texture_storage_3d<rgba32float, write>;

const RESOLUTION: f32 = 128.0;

////////////////////////////////////////////////////////////////
// MAIN COMPUTE SHADER
////////////////////////////////////////////////////////////////

@compute @workgroup_size(8, 8, 8)
fn generate_cloud_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let resolution: f32 = 128.0;
    let pos: vec3<f32> = vec3<f32>(id) / resolution;

    // Generate base Perlin noise
    let perlin: f32 = perlin_fbm(pos, 5.0, 8u);

    // Worley noises at increasing frequencies
    let worley1: f32 = worley_fbm(pos, 4.0);
    let worley2: f32 = worley_fbm(pos, 8.0);
    let worley3: f32 = worley_fbm(pos, 16.0);
    let worley4: f32 = worley_fbm(pos, 32.0);

    // Perlin-Worley noise
    let val1: f32 = clamp(perlin, worley1, 1.0);
    let perlin_worley: f32 = mix(worley1, 1.0, val1);

    let color = vec4<f32>(perlin, worley2, worley3, worley4);
    textureStore(output, vec3<i32>(id), color);
}

////////////////////////////////////////////////////////////////
// HIGH-LEVEL FBM NOISE FUNCTIONS
////////////////////////////////////////////////////////////////

// Fractional Brownian Motion (FBM) sums multiple octaves of noise.
fn perlin_fbm(pos: vec3<f32>, base_frequency: f32, octaves: u32) -> f32 {
    let persistance = 1.0;

    var frequency = base_frequency;
    var amplitude = 1.0;
    var max_value = 0.0;
    let repeat = RESOLUTION / 10.0;

    // Sum octaves of Perlin noise.
    var total = 0.0;
    for (var i = 0u; i < octaves; i++) {
        total += perlin(pos * frequency, repeat * frequency) * amplitude;

        max_value += amplitude;
        amplitude *= persistance;
        frequency *= 2.0;
    }

    // Normalize and clamp the final value.
    total /= max_value;
    return clamp(total, 0.0, 1.0);
}

fn worley_fbm(p: vec3<f32>, freq: f32) -> f32 {
    let o1 = worley(p * freq, freq) * 0.625;
    let o2 = worley(p * freq * 2.0, freq * 2.0) * 0.25;
    let o3 = worley(p * freq * 4.0, freq * 4.0) * 0.125;
    return pow(o1 + o2 + o3, 2.0);
}

////////////////////////////////////////////////////////////////
// Noise Functions
////////////////////////////////////////////////////////////////

// Quintic interpolation function.
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Calculates the dot product between a gradient vector and a point vector.
fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    var u = select(y, x, h < 8);
    var v = select(z, y, h < 4);
    v = select(v, x, h == 12 || h == 14);

    u = select(-u, u, (h & 1) == 0);
    v = select(-v, v, (h & 2) == 0);

    return u + v;
}

// White noise integer hash: vec3i → i32 [-1,1]
fn hash3i(p: vec3<i32>) -> i32 {
    var n: i32 = p.x * 374761393 + p.y * 668265263 + p.z * -1028477379;
    n = (n ^ (n >> 13)) * 1274126177;
    return n;
}

// Increments a coordinate, wrapping it based on the repeat value.
fn inc(val: i32, repeat: i32) -> i32 {
    return (val + 1) % repeat;
}

// 3D Perlin noise function.
fn perlin(pos: vec3<f32>, repeat: f32) -> f32 {
    // Repeat the coordinates.
    let pm = mod3(pos, repeat);

    // Integer cell coords, cast each component.
    let xi = i32(trunc(pm.x)) & 255;
    let yi = i32(trunc(pm.y)) & 255;
    let zi = i32(trunc(pm.z)) & 255;

    // Fractional part.
    let xf = fract(pm.x);
    let yf = fract(pm.y);
    let zf = fract(pm.z);

    // Apply fade function to fractional parts for smoother interpolation.
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Determine the 8 gradient vector hashes for the 8 corner points of the current unit cube
    let r = i32(repeat);
    let aaa = hash3i(vec3<i32>(xi, yi, zi));
    let aba = hash3i(vec3<i32>(xi, inc(yi, r), zi));
    let aab = hash3i(vec3<i32>(xi, yi, inc(zi, r)));
    let abb = hash3i(vec3<i32>(xi, inc(yi, r), inc(zi, r)));
    let baa = hash3i(vec3<i32>(inc(xi, r), yi, zi));
    let bba = hash3i(vec3<i32>(inc(xi, r), inc(yi, r), zi));
    let bab = hash3i(vec3<i32>(inc(xi, r), yi, inc(zi, r)));
    let bbb = hash3i(vec3<i32>(inc(xi, r), inc(yi, r), inc(zi, r)));

    // Bilinear interpolation in the x-y plane for the two sets of four points.
    let x1 = mix(grad(aaa, xf, yf, zf), grad(baa, xf - 1.0, yf, zf), u);
    let x2 = mix(grad(aba, xf, yf - 1.0, zf), grad(bba, xf - 1.0, yf - 1.0, zf), u);
    let y1 = mix(x1, x2, v);
    let x3 = mix(grad(aab, xf, yf, zf - 1.0), grad(bab, xf - 1.0, yf, zf - 1.0), u);
    let x4 = mix(grad(abb, xf, yf - 1.0, zf - 1.0), grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0), u);
    let y2 = mix(x3, x4, v);

    // Trilinear interpolation in z-direction for the resulting values, result to [0, 1] range.
    let res = (mix(y1, y2, w) + 1.0) / 2.0;

    return res;
}

// Cellular noise
fn worley(p: vec3<f32>, freq: f32) -> f32 {
    let cell: vec3<f32> = floor(p);
    let fractal: vec3<f32> = fract(p);
    var min_dist: f32 = 1e4;

    for (var x = -1.0; x <= 1.0; x += 1.0) {
        for (var y = -1.0; y <= 1.0; y += 1.0) {
            for (var z = -1.0; z <= 1.0; z += 1.0) {
                let offset: vec3<f32> = vec3<f32>(x, y, z);
                let point: vec3<f32> = hash33(mod3(cell + offset, freq)) * 0.5 + 0.5 + offset;
                let dist = fractal - point;
                min_dist = min(min_dist, dot(dist, dist));
            }
        }
    }

    return 1.0 - min_dist;
}

// Modulo function: vec3, f32 → vec3
fn mod3(x: vec3<f32>, y: f32) -> vec3<f32> {
    return x - floor(x / y) * y;
}

// White noise integer hash: vec3 → vec3 [-1,1]
const UI3: vec3<u32> = vec3<u32>(1597334673u, 3812015801u, 2798796415u);
const UIF: f32 = 1.0 / f32(0xffffffffu);
fn hash33(p: vec3<f32>) -> vec3<f32> {
    var qi: vec3<u32> = vec3<u32>(vec3<i32>(p)) * UI3;
    var q2: vec3<u32> = vec3<u32>(qi.x ^ qi.y ^ qi.z) * UI3;
    return vec3<f32>(q2) * UIF * 2.0 - 1.0;
}
