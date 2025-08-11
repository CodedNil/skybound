use bevy::math::{IVec3, Vec2, Vec3A};
use rayon::prelude::*;

// Generate a 3D Simplex Noise Texture
pub fn simplex_3d(
    size: usize,
    depth: usize,
    octaves: usize,
    gain: f32,
    freq: Vec2,
    gamma: f32,
) -> Vec<f32> {
    let freq = Vec3A::new(freq.x, freq.x, freq.y);
    (0..(size * size * depth))
        .into_par_iter()
        .map(|i| {
            // Unravel i into x,y,z
            let x = i % size;
            let y = (i / size) % size;
            let z = i / (size * size);

            // Normalized coordinates in [0..1]
            let pos = Vec3A::new(
                x as f32 / size as f32,
                y as f32 / size as f32,
                z as f32 / depth as f32,
            );

            // Compute fractal‐brownian motion
            let v = simplex_fbm3(pos, octaves, freq, gain);

            // Map from [-1..1] to [0..1]
            (v * 0.5 + 0.5).powf(gamma)
        })
        .collect()
}

// FBM Simplex Noise
fn simplex_fbm3(pos: Vec3A, octaves: usize, mut freq: Vec3A, gain: f32) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;

    for _ in 0..octaves {
        total += simplex3_seamless(pos * freq, freq) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }

    total / norm
}

fn simplex3_seamless(pos: Vec3A, period: Vec3A) -> f32 {
    let fpos = (pos % period) / period;
    let wx = fpos.x;
    let wy = fpos.y;
    let wz = fpos.z;

    let p000 = simplex3(pos);
    let p100 = simplex3(pos - Vec3A::new(period.x, 0.0, 0.0));
    let p010 = simplex3(pos - Vec3A::new(0.0, period.y, 0.0));
    let p110 = simplex3(pos - Vec3A::new(period.x, period.y, 0.0));

    let p001 = simplex3(pos - Vec3A::new(0.0, 0.0, period.z));
    let p101 = simplex3(pos - Vec3A::new(period.x, 0.0, period.z));
    let p011 = simplex3(pos - Vec3A::new(0.0, period.y, period.z));
    let p111 = simplex3(pos - Vec3A::new(period.x, period.y, period.z));

    let w000 = (1.0 - wx) * (1.0 - wy) * (1.0 - wz);
    let w100 = wx * (1.0 - wy) * (1.0 - wz);
    let w010 = (1.0 - wx) * wy * (1.0 - wz);
    let w110 = wx * wy * (1.0 - wz);

    let w001 = (1.0 - wx) * (1.0 - wy) * wz;
    let w101 = wx * (1.0 - wy) * wz;
    let w011 = (1.0 - wx) * wy * wz;
    let w111 = wx * wy * wz;

    p000 * w000
        + p100 * w100
        + p010 * w010
        + p110 * w110
        + p001 * w001
        + p101 * w101
        + p011 * w011
        + p111 * w111
}

// https://github.com/Razaekel/noise-rs/blob/develop/src/core/super_simplex.rs
const TO_SIMPLEX_CONSTANT_3D: f32 = -2.0 / 3.0;
const NORM_CONSTANT_3D: f32 = 1.0 / 0.086_766_4;

#[rustfmt::skip]
const LATTICE_LOOKUP_3D: [IVec3; 4 * 16] = [
    IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(0, 0, 1),
    IVec3::new(1, 1, 1), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(0, 0, 1),
    IVec3::new(0, 0, 0), IVec3::new(0, 1, 1), IVec3::new(0, 1, 0), IVec3::new(0, 0, 1),
    IVec3::new(1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(0, 1, 0), IVec3::new(0, 0, 1),
    IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), IVec3::new(0, 0, 1),
    IVec3::new(1, 1, 1), IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), IVec3::new(0, 0, 1),
    IVec3::new(0, 0, 0), IVec3::new(0, 1, 1), IVec3::new(1, 0, 1), IVec3::new(0, 0, 1),
    IVec3::new(1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(1, 0, 1), IVec3::new(0, 0, 1),
    IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 0), IVec3::new(0, 1, 1), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1), IVec3::new(1, 0, 0), IVec3::new(1, 0, 1), IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 0), IVec3::new(0, 1, 1), IVec3::new(1, 0, 1), IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(1, 0, 1), IVec3::new(1, 1, 0),
];

fn simplex3(point: Vec3A) -> f32 {
    // Transform point from real space to simplex space
    let to_simplex_offset = (point.x + point.y + point.z) * TO_SIMPLEX_CONSTANT_3D;
    let simplex_point = -(point + to_simplex_offset);
    let second_simplex_point = simplex_point + 512.5;

    // Get base point of simplex and barycentric coordinates in simplex space
    let simplex_base_point_i = simplex_point.floor().as_ivec3();
    let simplex_base_point = simplex_base_point_i.as_vec3a();
    let simplex_rel_coords = simplex_point - simplex_base_point;
    let second_simplex_base_point_i = second_simplex_point.floor().as_ivec3();
    let second_simplex_base_point = second_simplex_base_point_i.as_vec3a();
    let second_simplex_rel_coords = second_simplex_point - second_simplex_base_point;

    // Create indices to lookup table from barycentric coordinates
    let index =
        usize::from(simplex_rel_coords.x + simplex_rel_coords.y + simplex_rel_coords.z >= 1.5) << 2
            | usize::from(
                -simplex_rel_coords.x + simplex_rel_coords.y + simplex_rel_coords.z >= 0.5,
            ) << 3
            | usize::from(
                simplex_rel_coords.x - simplex_rel_coords.y + simplex_rel_coords.z >= 0.5,
            ) << 4
            | usize::from(
                simplex_rel_coords.x + simplex_rel_coords.y - simplex_rel_coords.z >= 0.5,
            ) << 5;
    let second_index = usize::from(
        second_simplex_rel_coords.x + second_simplex_rel_coords.y + second_simplex_rel_coords.z
            >= 1.5,
    ) << 2
        | usize::from(
            -second_simplex_rel_coords.x
                + second_simplex_rel_coords.y
                + second_simplex_rel_coords.z
                >= 0.5,
        ) << 3
        | usize::from(
            second_simplex_rel_coords.x - second_simplex_rel_coords.y + second_simplex_rel_coords.z
                >= 0.5,
        ) << 4
        | usize::from(
            second_simplex_rel_coords.x + second_simplex_rel_coords.y - second_simplex_rel_coords.z
                >= 0.5,
        ) << 5;

    let mut value = 0.0;

    // Sum contributions from first lattice
    for &lattice_lookup in &LATTICE_LOOKUP_3D[index..index + 4] {
        let dpos = simplex_rel_coords - lattice_lookup.as_vec3a();
        let attn = 0.75 - dpos.length_squared();
        if attn > 0.0 {
            let lattice_point = simplex_base_point_i + lattice_lookup;
            let gradient = Vec3A::from(grad3(hash3(lattice_point)));
            value += attn.powi(4) * gradient.dot(dpos);
        }
    }

    // Sum contributions from second lattice
    for &lattice_lookup in &LATTICE_LOOKUP_3D[second_index..second_index + 4] {
        let dpos = second_simplex_rel_coords - lattice_lookup.as_vec3a();
        let attn = 0.75 - dpos.length_squared();
        if attn > 0.0 {
            let lattice_point = second_simplex_base_point_i + lattice_lookup;
            let gradient = Vec3A::from(grad3(hash3(lattice_point)));
            value += attn.powi(4) * gradient.dot(dpos);
        }
    }

    value * NORM_CONSTANT_3D
}

#[rustfmt::skip]
fn grad3(index: usize) -> [f32; 3] {
    // Vectors are combinations of -1, 0, and 1
    // Precompute the normalized elements
    const DIAG : f32 = core::f32::consts::FRAC_1_SQRT_2;
    const DIAG2 : f32 = 0.577_350_26;

    match index % 32 {
        // 12 edges repeated twice then 8 corners
        0  | 12 => [  DIAG,   DIAG,    0.0],
        1  | 13 => [ -DIAG,   DIAG,    0.0],
        2  | 14 => [  DIAG,  -DIAG,    0.0],
        3  | 15 => [ -DIAG,  -DIAG,    0.0],
        4  | 16 => [  DIAG,    0.0,   DIAG],
        5  | 17 => [ -DIAG,    0.0,   DIAG],
        6  | 18 => [  DIAG,    0.0,  -DIAG],
        7  | 19 => [ -DIAG,    0.0,  -DIAG],
        8  | 20 => [   0.0,   DIAG,   DIAG],
        9  | 21 => [   0.0,  -DIAG,   DIAG],
        10 | 22 => [   0.0,   DIAG,  -DIAG],
        11 | 23 => [   0.0,  -DIAG,  -DIAG],
        24      => [ DIAG2,  DIAG2,  DIAG2],
        25      => [-DIAG2,  DIAG2,  DIAG2],
        26      => [ DIAG2, -DIAG2,  DIAG2],
        27      => [-DIAG2, -DIAG2,  DIAG2],
        28      => [ DIAG2,  DIAG2, -DIAG2],
        29      => [-DIAG2,  DIAG2, -DIAG2],
        30      => [ DIAG2, -DIAG2, -DIAG2],
        31      => [-DIAG2, -DIAG2, -DIAG2],
        _       => panic!("Attempt to access gradient {} of 32", index % 32),
    }
}

// 3D hashing function
const HASH_MULTIPLIER: f32 = 1.0 / 268_435_455.0;
fn hash3(p: IVec3) -> usize {
    let mut n: i32 = p.x * 3 + p.y * 113 + p.z * 311;
    n = ((n << 13) ^ n)
        .wrapping_mul(n.wrapping_mul(n).wrapping_mul(15731).wrapping_add(789_221))
        .wrapping_add(1_376_312_589);
    ((n & 268_435_455) as f32 * HASH_MULTIPLIER * 255.0) as usize
}
