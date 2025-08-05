use bevy::math::Vec3;
use orx_parallel::*;

// FBM Perlin Noise
pub fn worley_octave_3d(
    width: usize,
    height: usize,
    depth: usize,
    res: usize,
    pow: f32,
) -> Vec<u8> {
    // Generate three octaves of noise
    worley_3d(width, height, depth, res)
        .iter()
        .zip(worley_3d(width, height, depth, res * 2).iter())
        .zip(worley_3d(width, height, depth, res * 4).iter())
        .map(|((&o1, &o2), &o3)| {
            let combined_value = o1 * 0.625 + o2 * 0.25 + o3 * 0.125;
            combined_value.powf(pow)
        })
        .map(|v| (v * 255.0).round() as u8)
        .collect()
}

/// Generate 3D Worley (cellular) noise.
pub fn worley_3d(width: usize, height: usize, depth: usize, freq: usize) -> Vec<f32> {
    // size of a cell in world‐space
    let inv_cell = Vec3::new(
        freq as f32 / width as f32,
        freq as f32 / height as f32,
        freq as f32 / depth as f32,
    );
    let tile = freq as i32;
    let plane = width * height;
    let total = plane * depth;

    // Flatten the 3D loops into one big parallel iterator
    (0..total)
        .into_par()
        .map(|i| {
            // Unravel i -> (x,y,z)
            let z = i / plane;
            let rem = i % plane;
            let y = rem / width;
            let x = rem % width;

            // Point in (0..freq) space
            let p = (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * inv_cell;
            let cell = p.floor();
            let frac = p - cell;
            let ci = cell.x as i32;
            let cj = cell.y as i32;
            let ck = cell.z as i32;

            // Search the 3×3×3 neighborhood for the closest feature-point
            let mut best_d2 = f32::MAX;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = (ci + dx).rem_euclid(tile) as u32;
                        let ny = (cj + dy).rem_euclid(tile) as u32;
                        let nz = (ck + dz).rem_euclid(tile) as u32;

                        // Very cheap hash → three pseudo-random offsets in [0,1)
                        let mut h = nx.wrapping_mul(73856093)
                            ^ ny.wrapping_mul(19349669)
                            ^ nz.wrapping_mul(83492791);
                        h ^= (h >> 13) ^ (h << 17);
                        let fx = ((h & 0xFF) as f32) / 255.0;
                        let fy = (((h >> 8) & 0xFF) as f32) / 255.0;
                        let fz = (((h >> 16) & 0xFF) as f32) / 255.0;

                        let ft = Vec3::new(fx + dx as f32, fy + dy as f32, fz + dz as f32);
                        let d2 = (ft - frac).dot(ft - frac);
                        best_d2 = best_d2.min(d2);
                    }
                }
            }

            // Final distance→[1..0] value
            (1.0 - best_d2.sqrt()).clamp(0.0, 1.0)
        })
        .collect()
}
