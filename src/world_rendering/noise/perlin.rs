// Generate a 3D Perlin Noise Texture
pub fn perlin_image_3d(
    width: usize,
    height: usize,
    depth: usize,
    res: usize,
    persistance: usize,
) -> Vec<f32> {
    let mut data = Vec::with_capacity(width * height * depth);
    for x in 0..width {
        for y in 0..height {
            for z in 0..depth {
                // Normalize x,y,z to [0.0, 1.0]
                let nx = x as f32 / (width - 1) as f32;
                let ny = y as f32 / (height - 1) as f32;
                let nz = z as f32 / (depth - 1) as f32;

                // Sample noise in [-1.0..+1.0]
                let v = octave_3d(nx, ny, nz, 4, res, persistance as f32);

                // Remap from [-1,1] → [0,1] → [0,255]
                data.push(v * 0.5 + 0.5);
            }
        }
    }
    data
}

// FBM Perlin Noise
fn octave_3d(x: f32, y: f32, z: f32, repeat: usize, octaves: usize, persistence: f32) -> f32 {
    let mut total = 0.0;
    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        total += perlin_3d(x * frequency, y * frequency, z * frequency, repeat) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    total / max_value
}

// Single‐frequency Perlin
fn perlin_3d(mut x: f32, mut y: f32, mut z: f32, repeat: usize) -> f32 {
    // Apply repeat‐period wrap‐around if needed
    if repeat > 0 {
        x = x.rem_euclid(repeat as f32);
        y = y.rem_euclid(repeat as f32);
        z = z.rem_euclid(repeat as f32);
    }

    // Integer part
    let xi = (x.floor() as usize) & 255;
    let yi = (y.floor() as usize) & 255;
    let zi = (z.floor() as usize) & 255;

    // Fractional part
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    // Fade curves
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Get offset integers
    let xib = inc(xi, repeat);
    let yib = inc(yi, repeat);
    let zib = inc(zi, repeat);

    // Hash coordinates of the 8 cube corners
    let aaa = hash_3d(xi, yi, zi);
    let aba = hash_3d(xi, yib, zi);
    let aab = hash_3d(xi, yi, zib);
    let abb = hash_3d(xi, yib, zib);
    let baa = hash_3d(xib, yi, zi);
    let bba = hash_3d(xib, yib, zi);
    let bab = hash_3d(xib, yi, zib);
    let bbb = hash_3d(xib, yib, zib);

    // Interpolate
    let x1 = lerp(grad_3d(aaa, xf, yf, zf), grad_3d(baa, xf - 1.0, yf, zf), u);
    let x2 = lerp(
        grad_3d(aba, xf, yf - 1.0, zf),
        grad_3d(bba, xf - 1.0, yf - 1.0, zf),
        u,
    );
    let y1 = lerp(x1, x2, v);

    let x1 = lerp(
        grad_3d(aab, xf, yf, zf - 1.0),
        grad_3d(bab, xf - 1.0, yf, zf - 1.0),
        u,
    );
    let x2 = lerp(
        grad_3d(abb, xf, yf - 1.0, zf - 1.0),
        grad_3d(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
        u,
    );
    let y2 = lerp(x1, x2, v);

    (lerp(y1, y2, w) + 1.0) / 2.0
}

// Hash using perm table lookup
const PERM: [usize; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];
fn hash_3d(x: usize, y: usize, z: usize) -> usize {
    return PERM[PERM[PERM[x & 255] + y & 255] + z & 255];
}

// Increment with wrap if repeat > 0
#[inline]
fn inc(num: usize, repeat: usize) -> usize {
    let mut n = num + 1;
    if repeat > 0 {
        n %= repeat;
    }
    n
}

// Fade function 6t^5 − 15t^4 + 10t^3
#[inline]
fn fade(t: f32) -> f32 {
    // t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    let t2 = t * t;
    let t3 = t2 * t;
    t3 * (t * (t * 6.0 - 15.0) + 10.0)
}

// Linear interpolation
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// Gradient function
#[inline]
fn grad_3d(hash: usize, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    // pick u from x,y
    let mut u = if h < 8 { x } else { y };
    if (h & 1) != 0 {
        u = -u;
    }
    // pick v from y,z,x
    let mut v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    if (h & 2) != 0 {
        v = -v;
    }
    u + v
}
