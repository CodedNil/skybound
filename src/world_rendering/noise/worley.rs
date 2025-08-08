use bevy::math::Vec3A;
use orx_parallel::*;

// FBM Worley Noise
pub fn worley_3d_fbm(size: usize, depth: usize, freq: f32, gamma: f32) -> Vec<f32> {
    (0..(size * size * depth))
        .par()
        .map(|i| {
            // Unravel i -> (x,y,z)
            let x = i % size;
            let y = (i / size) % size;
            let z = i / (size * size);

            // Normalized coordinates in [0..1]
            let pos = Vec3A::new(
                x as f32 / size as f32,
                y as f32 / size as f32,
                z as f32 / depth as f32,
            );

            let o1 = worley3(pos, freq);
            let o2 = worley3(pos, freq * 2.0);
            let o3 = worley3(pos, freq * 4.0);

            (o1 * 0.625 + o2 * 0.25 + o3 * 0.125).powf(gamma)
        })
        .collect()
}

// FBM Worley Noise
pub fn worley_3d(size: usize, depth: usize, freq: f32, gamma: f32) -> Vec<f32> {
    (0..(size * size * depth))
        .par()
        .map(|i| {
            // Unravel i -> (x,y,z)
            let x = i % size;
            let y = (i / size) % size;
            let z = i / (size * size);

            // Normalized coordinates in [0..1]
            let pos = Vec3A::new(
                x as f32 / size as f32,
                y as f32 / size as f32,
                z as f32 / depth as f32,
            );

            worley3(pos, freq).powf(gamma)
        })
        .collect()
}

/// Generate 3D Worley (cellular) noise.
fn worley3(pos: Vec3A, freq: f32) -> f32 {
    let tile = freq as i32;

    // Point in cell‐space
    let fx0 = pos.x * freq;
    let fy0 = pos.y * freq;
    let fz0 = pos.z * freq;

    // Cell corner
    let cx = fx0.floor() as i32;
    let cy = fy0.floor() as i32;
    let cz = fz0.floor() as i32;

    // Fractional part inside the cell
    let dx0 = fx0 - cx as f32;
    let dy0 = fy0 - cy as f32;
    let dz0 = fz0 - cz as f32;

    // Search the 3×3×3 neighbors
    let mut best = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                // Wrap cell coords
                let nx = (cx + dx).rem_euclid(tile) as u32;
                let ny = (cy + dy).rem_euclid(tile) as u32;
                let nz = (cz + dz).rem_euclid(tile) as u32;

                // Cheap integer hash → 3 bytes of randomness
                let mut h = nx.wrapping_mul(73856093)
                    ^ ny.wrapping_mul(19349669)
                    ^ nz.wrapping_mul(83492791);
                h ^= (h >> 13) ^ (h << 17);

                // Unpack into [0..1)
                let ox = ((h & 0xFF) as f32) * (1.0 / 255.0);
                let oy = (((h >> 8) & 0xFF) as f32) * (1.0 / 255.0);
                let oz = (((h >> 16) & 0xFF) as f32) * (1.0 / 255.0);

                // Distance squared to feature point
                let dx1 = ox + dx as f32 - dx0;
                let dy1 = oy + dy as f32 - dy0;
                let dz1 = oz + dz as f32 - dz0;
                let d2 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

                if d2 < best {
                    best = d2;
                }
            }
        }
    }

    // Final distance→[1..0] value
    (1.0 - best.sqrt()).clamp(0.0, 1.0)
}
