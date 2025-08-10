use crate::world_rendering::noise::perlin::perlin_fbm3;
use bevy::math::Vec3A;
use rayon::prelude::*;

/// Computes the 2D curl magnitude of a scalar field
const EPSILON: f32 = 0.001;
const SCALE: f32 = 0.005;
const OCTAVES: usize = 5;
const FREQUENCY: Vec3A = Vec3A::new(2.0, 2.0, 2.0);
const GAIN: f32 = 0.4;
const GAMMA: f32 = 1.2;
pub fn curl_2d_texture(width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_pixels = width * height;

    let curl_values: Vec<Vec3A> = (0..total_pixels)
        .into_par_iter()
        .map(|i| {
            let x = i % width;
            let y = i / width;

            // Map pixel coordinates to noise space
            let x_f = (x as f32) * SCALE;
            let y_f = (y as f32) * SCALE;

            // Calculate the 6 noise values using a clean, vector-based syntax
            let pos = Vec3A::new(x_f, y_f, 0.0);
            let noise_x_plus =
                perlin_fbm3(pos + Vec3A::X * EPSILON, OCTAVES, FREQUENCY, GAIN, false);
            let noise_x_minus =
                perlin_fbm3(pos - Vec3A::X * EPSILON, OCTAVES, FREQUENCY, GAIN, false);
            let noise_y_plus =
                perlin_fbm3(pos + Vec3A::Y * EPSILON, OCTAVES, FREQUENCY, GAIN, false);
            let noise_y_minus =
                perlin_fbm3(pos - Vec3A::Y * EPSILON, OCTAVES, FREQUENCY, GAIN, false);
            let noise_z_plus =
                perlin_fbm3(pos + Vec3A::Z * EPSILON, OCTAVES, FREQUENCY, GAIN, false);
            let noise_z_minus =
                perlin_fbm3(pos - Vec3A::Z * EPSILON, OCTAVES, FREQUENCY, GAIN, false);

            // Calculate approximate partial derivatives.
            let d_noise_dx = (noise_x_plus - noise_x_minus) / (2.0 * EPSILON);
            let d_noise_dy = (noise_y_plus - noise_y_minus) / (2.0 * EPSILON);
            let d_noise_dz = (noise_z_plus - noise_z_minus) / (2.0 * EPSILON);

            // Calculate the curl components.
            let curl_x = d_noise_dy - d_noise_dz;
            let curl_y = d_noise_dx + d_noise_dy;
            let curl_z = d_noise_dz - d_noise_dx;
            Vec3A::new(curl_x, curl_y, curl_z)
        })
        .collect();

    // Find the maximum absolute value among all curl components
    let max_curl_magnitude = curl_values.iter().fold(0.0f32, |acc, v| {
        acc.max(v.x.abs().max(v.y.abs()).max(v.z.abs()))
    });
    let max_curl_magnitude = if max_curl_magnitude == 0.0 {
        1.0 // Avoid division by zero
    } else {
        max_curl_magnitude
    };

    // Second pass: normalize and map the values to u8
    let mut r_data = Vec::with_capacity(total_pixels);
    let mut g_data = Vec::with_capacity(total_pixels);
    let mut b_data = Vec::with_capacity(total_pixels);
    for curl_vec in curl_values {
        // Normalize the curl components to the range [-1, 1]
        let normalized = curl_vec / max_curl_magnitude;

        // Normalize the values from the range [-1, 1] to [0, 255]
        let normalise_value = |v: f32| -> f32 {
            let v_mapped = v * 0.5 + 0.5;
            v_mapped.powf(GAMMA)
        };
        r_data.push(normalise_value(normalized.x));
        g_data.push(normalise_value(normalized.y));
        b_data.push(normalise_value(normalized.z));
    }

    (r_data, g_data, b_data)
}
