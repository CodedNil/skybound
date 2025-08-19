use rayon::prelude::*;

// FBM Worley Noise
pub fn worley_3d(
    size: usize,
    depth: usize,
    octaves: usize,
    persistence: f32,
    freq: f32,
    gamma: f32,
) -> Vec<f32> {
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

            let mut total_value = 0.0;
            let mut total_amplitude = 0.0;
            let mut current_freq = freq;
            let mut current_amplitude = 1.0;

            for _ in 0..octaves {
                total_value += worley3(x, y, z, current_freq) * current_amplitude;
                total_amplitude += current_amplitude;
                current_freq *= 2.0;
                current_amplitude *= persistence;
            }

            // Normalize the final value by the total amplitude
            let val = if total_amplitude > 0.0 {
                total_value / total_amplitude
            } else {
                0.0
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
    let mut h =
        nx.wrapping_mul(73_856_093) ^ ny.wrapping_mul(19_349_669) ^ nz.wrapping_mul(83_492_791);
    h ^= h >> 13;
    h = h.wrapping_mul(2_246_822_507); // Murmur-like mix

    let ox = (h & 0xFF) as f32 * (1.0 / 255.0);
    let oy = ((h >> 8) & 0xFF) as f32 * (1.0 / 255.0);
    let oz = ((h >> 16) & 0xFF) as f32 * (1.0 / 255.0);
    (ox, oy, oz)
}
