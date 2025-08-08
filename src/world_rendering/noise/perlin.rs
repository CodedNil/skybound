use bevy::math::Vec3A;
use orx_parallel::*;

// Generate a 3D Perlin Noise Texture
pub fn perlin_3d(
    width: usize,
    height: usize,
    depth: usize,
    octaves: usize,
    gain: f32,
    freq: f32,
    pow: f32,
) -> Vec<u8> {
    (0..depth)
        .par()
        .flat_map(|z| {
            let mut slice = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let pos = Vec3A::new(
                        x as f32 / width as f32,
                        y as f32 / height as f32,
                        z as f32 / depth as f32,
                    );

                    // Compute fractal‐brownian motion
                    let v = perlin_fbm3(pos, octaves, freq, gain, true);

                    // Map from [-1..1] to [0..255]
                    slice.push(((v * 0.5 + 0.5).powf(pow) * 255.0).round() as u8);
                }
            }
            slice
        })
        .collect()
}

// FBM Perlin Noise
pub fn perlin_fbm3(pos: Vec3A, octaves: usize, mut freq: f32, gain: f32, tile: bool) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;

    for _ in 0..octaves {
        total += perlin3(pos * freq, freq, tile) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }

    total / norm
}

// Single‐frequency Perlin
const OFF: [Vec3A; 8] = [
    Vec3A::new(0.0, 0.0, 0.0),
    Vec3A::new(1.0, 0.0, 0.0),
    Vec3A::new(0.0, 1.0, 0.0),
    Vec3A::new(1.0, 1.0, 0.0),
    Vec3A::new(0.0, 0.0, 1.0),
    Vec3A::new(1.0, 0.0, 1.0),
    Vec3A::new(0.0, 1.0, 1.0),
    Vec3A::new(1.0, 1.0, 1.0),
];
fn perlin3(pos: Vec3A, period: f32, tile: bool) -> f32 {
    // Cell corner + local coords
    let p = pos.floor();
    let w = pos.fract();

    // Quintic blend: u = w³·(w·(w·6−15)+10)
    let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    // Compute dot(grad, offset) for each corner
    let mut dots = [0.0; 8];
    for (idx, &off) in OFF.iter().enumerate() {
        // Tile the integer cell coordinates so that noise is periodic at 'period'
        let coords = if tile {
            (p + off).rem_euclid(Vec3A::splat(period))
        } else {
            p + off
        };
        let (x, y, z) = (coords.x as usize, coords.y as usize, coords.z as usize);

        // Permutation table lookups to get the gradient index
        let hash = P[P[P[x % 256] as usize + y % 256] as usize + z % 256];
        let grad = GRADIENTS[hash as usize & 15];

        let disp = w - off;
        dots[idx] = disp.dot(grad);
    }

    // Trilinearly interpolate
    let lerp = |a, b, t| a + (b - a) * t;

    // Along x
    let x00 = lerp(dots[0], dots[1], u.x);
    let x10 = lerp(dots[2], dots[3], u.x);
    let x01 = lerp(dots[4], dots[5], u.x);
    let x11 = lerp(dots[6], dots[7], u.x);

    // Along y
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);

    // Along z
    lerp(y0, y1, u.z)
}

// Permutation table
const P_TABLE: [u8; 256] = [
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
// Double the table for simplified indexing
const P: [u8; 512] = {
    let mut p = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        p[i] = P_TABLE[i];
        p[i + 256] = P_TABLE[i];
        i += 1;
    }
    p
};

// Standard gradient vectors for 3D noise
const GRADIENTS: [Vec3A; 16] = [
    Vec3A::new(1.0, 1.0, 0.0),
    Vec3A::new(-1.0, 1.0, 0.0),
    Vec3A::new(1.0, -1.0, 0.0),
    Vec3A::new(-1.0, -1.0, 0.0),
    Vec3A::new(1.0, 0.0, 1.0),
    Vec3A::new(-1.0, 0.0, 1.0),
    Vec3A::new(1.0, 0.0, -1.0),
    Vec3A::new(-1.0, 0.0, -1.0),
    Vec3A::new(0.0, 1.0, 1.0),
    Vec3A::new(0.0, -1.0, 1.0),
    Vec3A::new(0.0, 1.0, -1.0),
    Vec3A::new(0.0, -1.0, -1.0),
    Vec3A::new(1.0, 1.0, 0.0),
    Vec3A::new(-1.0, 1.0, 0.0),
    Vec3A::new(0.0, -1.0, 1.0),
    Vec3A::new(0.0, -1.0, -1.0),
];
