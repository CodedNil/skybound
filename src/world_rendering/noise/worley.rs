use bevy::prelude::*;
use rand::Rng;

// FBM Perlin Noise
pub fn worley_octave_3d(width: usize, height: usize, depth: usize, res: usize) -> Vec<f32> {
    // Generate three octaves of noise
    worley_3d(width, height, depth, res)
        .iter()
        .zip(worley_3d(width, height, depth, res * 2).iter())
        .zip(worley_3d(width, height, depth, res * 4).iter())
        .map(|((&o1, &o2), &o3)| {
            let combined_value = o1 * 0.625 + o2 * 0.25 + o3 * 0.125;
            combined_value.powf(0.3)
        })
        .collect()
}

/// Generate 3D Worley (cellular) noise.
pub fn worley_3d(width: usize, height: usize, depth: usize, res: usize) -> Vec<f32> {
    // Determine the size of each cell
    let cell_size = Vec3::new(
        width as f32 / res as f32,
        height as f32 / res as f32,
        depth as f32 / res as f32,
    );

    // Generate one random point per cell
    let mut rng = rand::rng();
    let mut points = Vec::with_capacity(res * res * res);
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let base = Vec3::new(x as f32, y as f32, z as f32) * cell_size;
                let jitter = Vec3::new(
                    rng.random_range(0.0..cell_size.x),
                    rng.random_range(0.0..cell_size.y),
                    rng.random_range(0.0..cell_size.z),
                );
                points.push(base + jitter);
            }
        }
    }

    let mut max_dist = 0.0f32;
    let mut distances: Vec<f32> = Vec::with_capacity(width * height * depth);

    // Calculate the shortest distance for each voxel
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let cell = (center / cell_size).floor().as_ivec3();

                // Search the 3x3x3 neighborhood of cells
                let mut min_d = f32::MAX;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let off = IVec3::new(dx, dy, dz);
                            let neigh = (cell + off).rem_euclid(IVec3::splat(res as i32));
                            let idx = (neigh.x
                                + neigh.y * res as i32
                                + neigh.z * res as i32 * res as i32)
                                as usize;

                            let wrap = IVec3::new(
                                (cell.x + dx).div_euclid(res as i32),
                                (cell.y + dy).div_euclid(res as i32),
                                (cell.z + dz).div_euclid(res as i32),
                            );
                            let world_off = Vec3::new(
                                wrap.x as f32 * width as f32,
                                wrap.y as f32 * height as f32,
                                wrap.z as f32 * depth as f32,
                            );
                            let p = points[idx] + world_off;
                            min_d = min_d.min((p - center).length());
                        }
                    }
                }

                max_dist = max_dist.max(min_d);
                distances.push(min_d);
            }
        }
    }

    // Normalize and invert the distances
    distances
        .into_iter()
        .map(|d| 1.0 - (d / max_dist).clamp(0.0, 1.0))
        .collect()
}
