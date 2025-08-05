use bevy::math::{UVec3, Vec3};
use orx_parallel::*;

// Generate a 3D Perlin Noise Texture
pub fn perlin_3d(
    width: usize,
    height: usize,
    depth: usize,
    octaves: usize,
    base_freq: f32,
) -> Vec<u8> {
    let total_voxels = width * height * depth;
    (0..total_voxels)
        .par()
        .map(|i| {
            // unravel i into x,y,z
            let xy_plane = height * depth;
            let x = i / xy_plane;
            let rem = i % xy_plane;
            let y = rem / depth;
            let z = rem % depth;

            // normalized coordinates in [0..1]
            let pos = Vec3::new(
                x as f32 / (width - 1) as f32,
                y as f32 / (height - 1) as f32,
                z as f32 / (depth - 1) as f32,
            );

            // compute fractal‐brownian motion
            let v = fbm3(pos, octaves, base_freq);

            // map from [-1..1] to [0..255]
            ((v * 0.5 + 0.5) * 255.0).round() as u8
        })
        .collect()
}

// Generate a 2D Perlin Noise Texture
pub fn perlin_2d(width: usize, height: usize, octaves: usize, base_freq: f32) -> Vec<u8> {
    let total_voxels = width * height;
    (0..total_voxels)
        .par()
        .map(|i| {
            // unravel i → (x,y)
            let y = i / width;
            let x = i % width;

            // normalized coordinates in [0..1]
            let pos = Vec3::new(
                x as f32 / (width - 1) as f32,
                y as f32 / (height - 1) as f32,
                0.0,
            );

            // compute fractal‐brownian motion
            let v = fbm3(pos, octaves, base_freq);

            // map from [-1..1] to [0..255]
            ((v * 0.5 + 0.5) * 255.0).round() as u8
        })
        .collect()
}

// FBM Perlin Noise
fn fbm3(pos: Vec3, octaves: usize, mut freq: f32) -> f32 {
    let gain = 2.0_f32.powf(-0.85);
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;

    for _ in 0..octaves {
        total += perlin3(pos * freq, freq) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }

    total / norm
}

// Single‐frequency Perlin
const OFF: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
];
fn perlin3(pos: Vec3, period: f32) -> f32 {
    // Cell corner + local coords
    let p = pos.floor();
    let w = pos.fract();

    // Quintic blend: u = w³·(w·(w·6−15)+10)
    let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    // Compute dot(grad, offset) for each corner
    let mut dots = [0.0; 8];
    for (idx, &off) in OFF.iter().enumerate() {
        // Tile the integer cell coordinates so that noise is periodic at 'period'
        let corner = Vec3::new(
            (p.x + off.x).rem_euclid(period),
            (p.y + off.y).rem_euclid(period),
            (p.z + off.z).rem_euclid(period),
        );
        let grad = hash33(corner);
        let disp = w - off;
        dots[idx] = grad.dot(disp);
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

// Simple 3D→3D hash to get a pseudo‐random gradient in [-1..1]
const UI3: UVec3 = UVec3::new(1597334673, 3812015801, 2798796415);
const UIF: f32 = 1.0 / (u32::MAX as f32);
fn hash33(p: Vec3) -> Vec3 {
    let ip = UVec3::new(p.x as u32, p.y as u32, p.z as u32);
    let q = UVec3::new(
        ip.x.wrapping_mul(UI3.x),
        ip.y.wrapping_mul(UI3.y),
        ip.z.wrapping_mul(UI3.z),
    );
    let r = q.x ^ q.y ^ q.z;
    let q2 = UVec3::new(
        r.wrapping_mul(UI3.x),
        r.wrapping_mul(UI3.y),
        r.wrapping_mul(UI3.z),
    );
    // scale into [0..1], then remap to [-1..1]
    let f = Vec3::new(q2.x as f32, q2.y as f32, q2.z as f32) * UIF;
    2.0 * f - Vec3::ONE
}
