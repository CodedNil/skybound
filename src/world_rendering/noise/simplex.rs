use bevy::math::{IVec2, IVec3, Vec2, Vec3};
use orx_parallel::*;

// Generate a 3D Simplex Noise Texture
pub fn simplex_3d(
    width: usize,
    height: usize,
    depth: usize,
    octaves: usize,
    gain: f32,
    freq: f32,
    pow: f32,
) -> Vec<u8> {
    let total_voxels = width * height * depth;
    (0..total_voxels)
        .par()
        .map(|i| {
            // Unravel i into x,y,z
            let x = i / (height * depth);
            let y = (i / depth) % height;
            let z = i % depth;

            // Normalized coordinates in [0..1]
            let pos = Vec3::new(
                x as f32 / (width - 1) as f32,
                y as f32 / (height - 1) as f32,
                z as f32 / (depth - 1) as f32,
            );

            // Compute fractal‐brownian motion
            let v = if depth == 1 {
                simplex_fbm2(pos.truncate(), octaves, freq, gain)
            } else {
                simplex_fbm3(pos, octaves, freq, gain, true)
            };

            // Map from [-1..1] to [0..255]
            ((v * 0.5 + 0.5).powf(pow) * 255.0).round() as u8
        })
        .collect()
}

// FBM Simplex Noise
#[inline(always)]
pub fn simplex_fbm3(pos: Vec3, octaves: usize, mut freq: f32, gain: f32, tile: bool) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;

    for _ in 0..octaves {
        total += simplex3(pos * freq, freq, tile) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }

    total / norm
}

#[inline(always)]
pub fn simplex_fbm2(pos: Vec2, octaves: usize, mut freq: f32, gain: f32) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;

    for _ in 0..octaves {
        total += simplex2(pos * freq) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }

    total / norm
}

// https://github.com/Razaekel/noise-rs/blob/develop/src/core/super_simplex.rs

const TO_REAL_CONSTANT_2D: f32 = -0.211_324_865_405_187; // (1 / sqrt(2 + 1) - 1) / 2
const TO_SIMPLEX_CONSTANT_2D: f32 = 0.366_025_403_784_439; // (sqrt(2 + 1) - 1) / 2
const TO_SIMPLEX_CONSTANT_3D: f32 = -2.0 / 3.0;

const NORM_CONSTANT_2D: f32 = 1.0 / 0.054_282_952_886_616_23;
const NORM_CONSTANT_3D: f32 = 1.0 / 0.086_766_400_165_536_9;

#[rustfmt::skip]
const LATTICE_LOOKUP_2D: [(IVec2, Vec2); 4 * 8] = [
    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(-1, 0), Vec2::new(0.788_675_134_594_813, -0.211_324_865_405_187)),
    (IVec2::new(0, -1), Vec2::new(-0.211_324_865_405_187, 0.788_675_134_594_813)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(0, 1), Vec2::new(0.211_324_865_405_187, -0.788_675_134_594_813)),
    (IVec2::new(1, 0), Vec2::new(-0.788_675_134_594_813, 0.211_324_865_405_187)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(1, 0), Vec2::new(-0.788_675_134_594_813, 0.211_324_865_405_187)),
    (IVec2::new(0, -1), Vec2::new(-0.211_324_865_405_187, 0.788_675_134_594_813)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(2, 1), Vec2::new(-1.366_025_403_784_439, -0.366_025_403_784_439)),
    (IVec2::new(1, 0), Vec2::new(-0.788_675_134_594_813, 0.211_324_865_405_187)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(-1, 0), Vec2::new(0.788_675_134_594_813, -0.211_324_865_405_187)),
    (IVec2::new(0, 1), Vec2::new(0.211_324_865_405_187, -0.788_675_134_594_813)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(0, 1), Vec2::new(0.211_324_865_405_187, -0.788_675_134_594_813)),
    (IVec2::new(1, 2), Vec2::new(-0.366_025_403_784_439, -1.366_025_403_784_439)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(1, 0), Vec2::new(-0.788_675_134_594_813, 0.211_324_865_405_187)),
    (IVec2::new(0, 1), Vec2::new(0.211_324_865_405_187, -0.788_675_134_594_813)),

    (IVec2::new(0, 0), Vec2::new(0.0, 0.0)),
    (IVec2::new(1, 1), Vec2::new(-0.577_350_269_189_626, -0.577_350_269_189_626)),
    (IVec2::new(2, 1), Vec2::new(-1.366_025_403_784_439, -0.366_025_403_784_439)),
    (IVec2::new(1, 2), Vec2::new(-0.366_025_403_784_439, -1.366_025_403_784_439)),
];

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

#[inline(always)]
pub fn simplex2(point: Vec2) -> f32 {
    // Transform point from real space to simplex space
    let to_simplex_offset = (point.x + point.y) * TO_SIMPLEX_CONSTANT_2D;
    let simplex_point = point + to_simplex_offset;

    // Get base point of simplex and barycentric coordinates in simplex space
    let simplex_base_point_i = simplex_point.floor().as_ivec2();
    let simplex_base_point = simplex_base_point_i.as_vec2();
    let simplex_rel_coords = simplex_point - simplex_base_point;

    // Create index to lookup table from barycentric coordinates
    let region_sum = (simplex_rel_coords.x + simplex_rel_coords.y).floor();
    let index = ((region_sum >= 1.0) as usize) << 2
        | ((simplex_rel_coords.x - simplex_rel_coords.y * 0.5 + 1.0 - region_sum * 0.5 >= 1.0)
            as usize)
            << 3
        | ((simplex_rel_coords.y - simplex_rel_coords.x * 0.5 + 1.0 - region_sum * 0.5 >= 1.0)
            as usize)
            << 4;

    // Transform barycentric coordinates to real space
    let to_real_offset = (simplex_rel_coords.x + simplex_rel_coords.y) * TO_REAL_CONSTANT_2D;
    let real_rel_coords = simplex_rel_coords + to_real_offset;

    let mut value = 0.0;

    for lattice_lookup in &LATTICE_LOOKUP_2D[index..index + 4] {
        let dpos = real_rel_coords + lattice_lookup.1;
        let attn = (2.0 / 3.0) - dpos.length_squared();
        if attn > 0.0 {
            let lattice_point = simplex_base_point_i + lattice_lookup.0;
            let gradient = Vec2::from(grad2(hash2(lattice_point)));
            value += attn.powi(4) * gradient.dot(dpos);
        }
    }

    value * NORM_CONSTANT_2D
}

#[inline(always)]
pub fn simplex3(point: Vec3, period: f32, tile: bool) -> f32 {
    // Transform point from real space to simplex space
    let to_simplex_offset = (point.x + point.y + point.z) * TO_SIMPLEX_CONSTANT_3D;
    let simplex_point = -(point + to_simplex_offset);
    let second_simplex_point = simplex_point + 512.5;

    // Get base point of simplex and barycentric coordinates in simplex space
    let simplex_base_point_i = simplex_point.floor().as_ivec3();
    let simplex_base_point = simplex_base_point_i.as_vec3();
    let simplex_rel_coords = simplex_point - simplex_base_point;
    let second_simplex_base_point_i = second_simplex_point.floor().as_ivec3();
    let second_simplex_base_point = second_simplex_base_point_i.as_vec3();
    let second_simplex_rel_coords = second_simplex_point - second_simplex_base_point;

    // Create indices to lookup table from barycentric coordinates
    let index = ((simplex_rel_coords.x + simplex_rel_coords.y + simplex_rel_coords.z >= 1.5)
        as usize)
        << 2
        | ((-simplex_rel_coords.x + simplex_rel_coords.y + simplex_rel_coords.z >= 0.5) as usize)
            << 3
        | ((simplex_rel_coords.x - simplex_rel_coords.y + simplex_rel_coords.z >= 0.5) as usize)
            << 4
        | ((simplex_rel_coords.x + simplex_rel_coords.y - simplex_rel_coords.z >= 0.5) as usize)
            << 5;
    let second_index = ((second_simplex_rel_coords.x
        + second_simplex_rel_coords.y
        + second_simplex_rel_coords.z
        >= 1.5) as usize)
        << 2
        | ((-second_simplex_rel_coords.x
            + second_simplex_rel_coords.y
            + second_simplex_rel_coords.z
            >= 0.5) as usize)
            << 3
        | ((second_simplex_rel_coords.x - second_simplex_rel_coords.y + second_simplex_rel_coords.z
            >= 0.5) as usize)
            << 4
        | ((second_simplex_rel_coords.x + second_simplex_rel_coords.y - second_simplex_rel_coords.z
            >= 0.5) as usize)
            << 5;

    let mut value = 0.0;

    // Sum contributions from first lattice
    for &lattice_lookup in &LATTICE_LOOKUP_3D[index..index + 4] {
        let dpos = simplex_rel_coords - lattice_lookup.as_vec3();
        let attn = 0.75 - dpos.length_squared();
        if attn > 0.0 {
            // Tile the integer cell coordinates so that noise is periodic at 'period'
            let lattice_point = simplex_base_point_i + lattice_lookup;
            let gradient = Vec3::from(grad3(hash3(lattice_point)));
            value += attn.powi(4) * gradient.dot(dpos);
        }
    }

    // Sum contributions from second lattice
    for &lattice_lookup in &LATTICE_LOOKUP_3D[second_index..second_index + 4] {
        let dpos = second_simplex_rel_coords - lattice_lookup.as_vec3();
        let attn = 0.75 - dpos.length_squared();
        if attn > 0.0 {
            // Tile the integer cell coordinates so that noise is periodic at 'period'
            let lattice_point = second_simplex_base_point_i + lattice_lookup;
            let gradient = Vec3::from(grad3(hash3(lattice_point)));
            value += attn.powi(4) * gradient.dot(dpos);
        }
    }

    value * NORM_CONSTANT_3D
}

#[rustfmt::skip]
#[inline(always)]
fn grad2(index: usize) -> [f32; 2] {
    // Vectors are combinations of -1, 0, and 1
    // Precompute the normalized element
    const DIAG : f32 = core::f32::consts::FRAC_1_SQRT_2;

    match index % 8 {
        0 => [  1.0,   0.0],
        1 => [ -1.0,   0.0],
        2 => [  0.0,   1.0],
        3 => [  0.0,  -1.0],
        4 => [ DIAG,  DIAG],
        5 => [-DIAG,  DIAG],
        6 => [ DIAG, -DIAG],
        7 => [-DIAG, -DIAG],
        _ => panic!("Attempt to access gradient {} of 8", index % 8),
    }
}

#[rustfmt::skip]
#[inline(always)]
fn grad3(index: usize) -> [f32; 3] {
    // Vectors are combinations of -1, 0, and 1
    // Precompute the normalized elements
    const DIAG : f32 = core::f32::consts::FRAC_1_SQRT_2;
    const DIAG2 : f32 = 0.577_350_269_189_625_8;

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

// 2D hashing function
#[inline(always)]
fn hash2(p: IVec2) -> usize {
    let mut n: i32 = p.x * 3 + p.y * 113;
    n = ((n << 13) ^ n)
        .wrapping_mul(n.wrapping_mul(n).wrapping_mul(15731).wrapping_add(789221))
        .wrapping_add(1376312589);

    ((n & 0x0fffffff) as f32 * HASH_MULTIPLIER * 255.0) as usize
}

// 3D hashing function
const HASH_MULTIPLIER: f32 = 1.0 / (0x0fffffff as f32);
#[inline(always)]
fn hash3(p: IVec3) -> usize {
    let mut n: i32 = p.x * 3 + p.y * 113 + p.z * 311;
    n = ((n << 13) ^ n)
        .wrapping_mul(n.wrapping_mul(n).wrapping_mul(15731).wrapping_add(789221))
        .wrapping_add(1376312589);
    ((n & 0x0fffffff) as f32 * HASH_MULTIPLIER * 255.0) as usize
}
