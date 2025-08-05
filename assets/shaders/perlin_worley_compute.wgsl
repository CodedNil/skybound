@group(0) @binding(0) var output: texture_storage_3d<rgba32float, write>;

const BASE_RESOLUTION: u32 = 128;

////////////////////////////////////////////////////////////////
// MAIN COMPUTE SHADER
////////////////////////////////////////////////////////////////

@compute @workgroup_size(8, 8, 8)
fn generate_cloud_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos: vec3<f32> = vec3<f32>(id) / f32(BASE_RESOLUTION);

    // Generate base Perlin noise
    let perlin: f32 = perlin_fbm(pos, 5u, 8.0);

    // Worley noises at increasing frequencies
    let worley1: f32 = worley_fbm(pos, 4.0);
    let worley2: f32 = worley_fbm(pos, 8.0);
    let worley3: f32 = worley_fbm(pos, 16.0);
    let worley4: f32 = worley_fbm(pos, 32.0);

    // Perlin-Worley noise
    let perlin_worley: f32 = mix(worley1, 1.0, clamp(perlin, worley1, 1.0));

    let color = vec4<f32>(perlin_worley, worley2, worley3, worley4);
    textureStore(output, vec3<i32>(id), color);
}

////////////////////////////////////////////////////////////////
// HIGH-LEVEL FBM NOISE FUNCTIONS
////////////////////////////////////////////////////////////////

// Fractional Brownian Motion (FBM) sums multiple octaves of noise.
fn perlin_fbm(p: vec3<f32>, octaves: u32, persistence: f32) -> f32 {
    var total = 0.0;
    var freq = 1.0;
    var amp = 1.0;
    var amplitude_sum = 0.0;
    for (var i = 0u; i < octaves; i++) {
        total += perlin(p * freq, 4) * amp;
        amplitude_sum += amp;
        amp *= persistence;
        freq *= 2.0;
    }
    return total / amplitude_sum;
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

// 3D Perlin noise function.
fn perlin(pos: vec3<f32>, repeat: i32) -> f32 {
    let p = mod3(pos, f32(repeat));

    let xi = i32(p.x) & 255;
    let yi = i32(p.y) & 255;
    let zi = i32(p.z) & 255;
    let xf = fract(p.x);
    let yf = fract(p.y);
    let zf = fract(p.z);

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let xib = inc(xi, repeat);
    let yib = inc(yi, repeat);
    let zib = inc(zi, repeat);

    // Determine the 8 gradient vector hashes for the 8 corner points of the current unit cube
    let aaa = hash3i(xi, yi, zi);
    let aba = hash3i(xi, yib, zi);
    let aab = hash3i(xi, yi, zib);
    let abb = hash3i(xi, yib, zib);
    let baa = hash3i(xib, yi, zi);
    let bba = hash3i(xib, yib, zi);
    let bab = hash3i(xib, yi, zib);
    let bbb = hash3i(xib, yib, zib);

    // Bilinear interpolation in the x-y plane for the two sets of four points.
    let x1 = mix(grad(aaa, xf, yf, zf), grad(baa, xf - 1.0, yf, zf), u);
    let x2 = mix(grad(aba, xf, yf - 1.0, zf), grad(bba, xf - 1.0, yf - 1.0, zf), u);
    let y1 = mix(x1, x2, v);

    let x3 = mix(grad(aab, xf, yf, zf - 1.0), grad(bab, xf - 1.0, yf, zf - 1.0), u);
    let x4 = mix(grad(abb, xf, yf - 1.0, zf - 1.0), grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0), u);
    let y2 = mix(x3, x4, v);

    // Trilinear interpolation in z-direction for the resulting values, result to [0, 1] range.
    return mix(y1, y2, w) * 0.5 + 0.5;
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

// Quintic interpolation function.
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Calculates the dot product between a gradient vector and a point vector.
fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    var u = select(x, y, h >= 8);
    if (h & 1) != 0 { u = -u; }
    var v = select(y, z, h < 4);
    if h == 12 { v = x; }
    if (h & 2) != 0 { v = -v; }
    return u + v;
}

// Hash using perm table lookup

const PERM: array<i32, 256> = array<i32, 256>(
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,
    120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,
    48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,
    41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
    187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
    186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,
    2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,
    110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,
    210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,
    181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,
    93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
);
fn hash3i(x: i32, y: i32, z: i32) -> i32 {
    return PERM[PERM[PERM[x & 255] + y & 255] + z & 255];
}

// Increments a coordinate, wrapping it based on the repeat value.
fn inc(val: i32, repeat: i32) -> i32 {
    return (val + 1) % repeat;
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
