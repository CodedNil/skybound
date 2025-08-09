use rayon::prelude::*;
// use std::simd::{
//     StdFloat,
//     cmp::{SimdPartialEq, SimdPartialOrd},
//     f32x8, i32x8,
//     num::{SimdFloat, SimdInt, SimdUint},
//     u32x8,
// };

// FBM Worley Noise
pub fn worley_3d(size: usize, depth: usize, freq: f32, gamma: f32, fbm: bool) -> Vec<f32> {
    let inv_size = 1.0 / size as f32;
    let inv_depth = 1.0 / depth as f32;

    (0..(size * size * depth))
        .into_par_iter()
        .map(|i| {
            // Unravel i -> (x,y,z) [0..1]
            let x = (i % size) as f32 * inv_size;
            let y = ((i / size) % size) as f32 * inv_size;
            let z = (i / (size * size)) as f32 * inv_depth;

            // Calculate Worley for each octave
            let val = if fbm {
                let o1 = worley3(x, y, z, freq);
                let o2 = worley3(x, y, z, freq * 2.0);
                let o3 = worley3(x, y, z, freq * 4.0);
                o1 * 0.625 + o2 * 0.25 + o3 * 0.125
            } else {
                worley3(x, y, z, freq)
            };
            val.powf(gamma)
        })
        .collect()
}

/// Generate 3D Worley (cellular) noise.
fn worley3(x: f32, y: f32, z: f32, freq: f32) -> f32 {
    let tile = freq as i32;

    // Point in cell‐space
    let fx0 = x * freq;
    let fy0 = y * freq;
    let fz0 = z * freq;

    // Cell corner
    let cx = fx0 as i32;
    let cy = fy0 as i32;
    let cz = fz0 as i32;

    // Fractional part inside the cell
    let dx0 = fx0 - cx as f32;
    let dy0 = fy0 - cy as f32;
    let dz0 = fz0 - cz as f32;

    // Search the 3×3×3 neighbors
    let mut best = f32::MAX;
    for &(dx, dy, dz) in &NEIGHBOURS {
        // Wrap cell coords and hash
        let nx = (cx + dx).rem_euclid(tile) as u32;
        let ny = (cy + dy).rem_euclid(tile) as u32;
        let nz = (cz + dz).rem_euclid(tile) as u32;
        let (ox, oy, oz) = hash3(nx, ny, nz);

        // Distance squared to feature point
        let dx1 = ox + dx as f32 - dx0;
        let dy1 = oy + dy as f32 - dy0;
        let dz1 = oz + dz as f32 - dz0;
        let d2 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

        if d2 < best {
            best = d2;
        }
    }

    // Final distance→[1..0] value
    (1.0 - best.sqrt()).clamp(0.0, 1.0)
}

const NEIGHBOURS: [(i32, i32, i32); 27] = {
    let mut arr = [(0, 0, 0); 27];
    let mut i = 0;
    let mut dz = -1;
    while dz <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                arr[i] = (dx, dy, dz);
                i += 1;
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    arr
};

#[inline(always)]
fn hash3(nx: u32, ny: u32, nz: u32) -> (f32, f32, f32) {
    let mut h = nx.wrapping_mul(73856093) ^ ny.wrapping_mul(19349669) ^ nz.wrapping_mul(83492791);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85ebca6b); // Murmur-like mix

    let ox = (h & 0xFF) as f32 * (1.0 / 255.0);
    let oy = ((h >> 8) & 0xFF) as f32 * (1.0 / 255.0);
    let oz = ((h >> 16) & 0xFF) as f32 * (1.0 / 255.0);
    (ox, oy, oz)
}

// const SIMD_CHANNELS: usize = 8;

// const INDICES_BASE_ARRAY: i32x8 = {
//     let mut indices = [0; SIMD_CHANNELS];
//     let mut i = 0;
//     while i < SIMD_CHANNELS {
//         indices[i] = i as i32;
//         i += 1;
//     }
//     i32x8::from_array(indices)
// };

// /// Create a 3D Worley noise texture with 3 octaves
// pub fn worley_3d(size: usize, depth: usize, freq: f32, gamma: f32, fbm: bool) -> Vec<f32> {
//     let len = size * size * depth;
//     let mut out = vec![0.0f32; len];

//     // Precompute constants
//     let freq1 = f32x8::splat(freq);
//     let freq2 = f32x8::splat(freq * 2.0);
//     let freq3 = f32x8::splat(freq * 4.0);

//     let tile1 = i32x8::splat(freq as i32);
//     let tile2 = i32x8::splat((freq * 2.0) as i32);
//     let tile3 = i32x8::splat((freq * 4.0) as i32);

//     let size_v = i32x8::splat(size as i32);
//     let size_sq_v = i32x8::splat((size * size) as i32);
//     let depth_v = i32x8::splat(depth as i32);
//     let inv_size = f32x8::splat(1.0 / size as f32);
//     let inv_depth = f32x8::splat(1.0 / depth as f32);

//     let w1 = f32x8::splat(0.625);
//     let w2 = f32x8::splat(0.25);
//     let w3 = f32x8::splat(0.125);
//     let gamma_v = f32x8::splat(gamma);

//     // Precompute neighbour offsets (both int and float)
//     let offsets_i = [
//         OFFSETS.map(|(dx, _, _)| i32x8::splat(dx)),
//         OFFSETS.map(|(_, dy, _)| i32x8::splat(dy)),
//         OFFSETS.map(|(_, _, dz)| i32x8::splat(dz)),
//     ];
//     let offsets_f = [
//         OFFSETS.map(|(dx, _, _)| f32x8::splat(dx as f32)),
//         OFFSETS.map(|(_, dy, _)| f32x8::splat(dy as f32)),
//         OFFSETS.map(|(_, _, dz)| f32x8::splat(dz as f32)),
//     ];

//     out.par_chunks_mut(SIMD_CHANNELS)
//         .enumerate()
//         .for_each(|(chunk_i, chunk)| {
//             let base_i = chunk_i * SIMD_CHANNELS;
//             let indices = i32x8::splat(base_i as i32) + INDICES_BASE_ARRAY;

//             let ix = simd_rem_euclid_i32(indices, size_v);
//             let iy = simd_rem_euclid_i32(indices / size_v, size_v);
//             let iz = simd_rem_euclid_i32(indices / size_sq_v, depth_v);

//             let pos_x = ix.cast::<f32>() * inv_size;
//             let pos_y = iy.cast::<f32>() * inv_size;
//             let pos_z = iz.cast::<f32>() * inv_depth;

//             // Calculate Worley for each octave
//             let val = if fbm {
//                 let o1 = simd_worley3(pos_x, pos_y, pos_z, &freq1, tile1, &offsets_i, &offsets_f);
//                 let o2 = simd_worley3(pos_x, pos_y, pos_z, &freq2, tile2, &offsets_i, &offsets_f);
//                 let o3 = simd_worley3(pos_x, pos_y, pos_z, &freq3, tile3, &offsets_i, &offsets_f);
//                 o1 * w1 + o2 * w2 + o3 * w3
//             } else {
//                 simd_worley3(pos_x, pos_y, pos_z, &freq1, tile1, &offsets_i, &offsets_f)
//             };
//             let val = simd_powf(val, gamma_v);

//             // Write to chunk
//             let arr = val.to_array();
//             let n = chunk.len();
//             chunk[..n].copy_from_slice(&arr[..n]);
//         });

//     out
// }

// /// Generate 3D Worley (cellular) noise.
// const SPLAT_ZERO: f32x8 = f32x8::splat(0.0);
// const SPLAT_ZEROI: i32x8 = i32x8::splat(0);
// const SPLAT_TINY: f32x8 = f32x8::splat(1e-6);
// const SPLAT_HALF: f32x8 = f32x8::splat(0.5);
// const SPLAT_ONE: f32x8 = f32x8::splat(1.0);
// const SPLAT_TWO: f32x8 = f32x8::splat(2.0);
// const SPLAT_MAX: f32x8 = f32x8::splat(f32::MAX);
// const SPLAT_255: u32x8 = u32x8::splat(255);
// const SPLAT_FACTOR: f32x8 = f32x8::splat(1.0 / 255.0);

// fn simd_worley3(
//     pos_x: f32x8,
//     pos_y: f32x8,
//     pos_z: f32x8,
//     freq: &f32x8,
//     tile_mask: i32x8,
//     offsets_i: &[[i32x8; 27]; 3],
//     offsets_f: &[[f32x8; 27]; 3],
// ) -> f32x8 {
//     let fx = pos_x * freq;
//     let fy = pos_y * freq;
//     let fz = pos_z * freq;

//     // Floor to get cell coords as i32
//     let fx_floor = fx.floor();
//     let fy_floor = fy.floor();
//     let fz_floor = fz.floor();
//     let cx = fx_floor.cast::<i32>();
//     let cy = fy_floor.cast::<i32>();
//     let cz = fz_floor.cast::<i32>();

//     // Fractional parts inside the cell
//     let dx0 = fx - fx_floor;
//     let dy0 = fy - fy_floor;
//     let dz0 = fz - fz_floor;

//     let mut best = SPLAT_MAX;
//     for i in 0..27 {
//         let nx_u = simd_rem_euclid_i32(cx + offsets_i[0][i], tile_mask).cast::<u32>();
//         let ny_u = simd_rem_euclid_i32(cy + offsets_i[1][i], tile_mask).cast::<u32>();
//         let nz_u = simd_rem_euclid_i32(cz + offsets_i[2][i], tile_mask).cast::<u32>();
//         let hash = simd_hash(nx_u, ny_u, nz_u);

//         let ox = ((hash & SPLAT_255).cast::<f32>()) * SPLAT_FACTOR;
//         let oy = (((hash >> 8) & SPLAT_255).cast::<f32>()) * SPLAT_FACTOR;
//         let oz = (((hash >> 16) & SPLAT_255).cast::<f32>()) * SPLAT_FACTOR;

//         let dx1 = ox + offsets_f[0][i] - dx0;
//         let dy1 = oy + offsets_f[1][i] - dy0;
//         let dz1 = oz + offsets_f[2][i] - dz0;

//         let dist2 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
//         best = best.simd_min(dist2);
//     }

//     (SPLAT_ONE - best.sqrt()).simd_clamp(SPLAT_ZERO, SPLAT_ONE)
// }

// #[rustfmt::skip]
// const OFFSETS: [(i32, i32, i32); 27] = [
//     (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
//     (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
//     (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),

//     (0, -1, -1), (0, -1, 0), (0, -1, 1),
//     (0, 0, -1), (0, 0, 0), (0, 0, 1),
//     (0, 1, -1), (0, 1, 0), (0, 1, 1),

//     (1, -1, -1), (1, -1, 0), (1, -1, 1),
//     (1, 0, -1), (1, 0, 0), (1, 0, 1),
//     (1, 1, -1), (1, 1, 0), (1, 1, 1),
// ];

// const SPLAT_HASH_A: u32x8 = u32x8::splat(73856093);
// const SPLAT_HASH_B: u32x8 = u32x8::splat(19349669);
// const SPLAT_HASH_C: u32x8 = u32x8::splat(83492791);

// #[inline(always)]
// fn simd_hash(nx: u32x8, ny: u32x8, nz: u32x8) -> u32x8 {
//     let mut h = nx * SPLAT_HASH_A ^ ny * SPLAT_HASH_B ^ nz * SPLAT_HASH_C;
//     h ^= h >> 13 ^ h << 17;
//     h
// }

// /// Fast SIMD powf approximation: x^y = exp(y * ln(x))
// #[inline(always)]
// fn simd_powf(base: f32x8, exp: f32x8) -> f32x8 {
//     if (exp.simd_eq(SPLAT_ONE)).all() {
//         return base;
//     }
//     if (exp.simd_eq(SPLAT_HALF)).all() {
//         return base.sqrt();
//     }
//     if (exp.simd_eq(SPLAT_TWO)).all() {
//         return base * base;
//     }
//     let base = base.simd_max(SPLAT_TINY);
//     (base.ln() * exp).exp()
// }

// #[inline(always)]
// fn simd_rem_euclid_i32(a: i32x8, m: i32x8) -> i32x8 {
//     let r = a % m;
//     r + (r.simd_lt(SPLAT_ZEROI)).select(m, SPLAT_ZEROI)
// }
