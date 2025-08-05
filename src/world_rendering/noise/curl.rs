use crate::world_rendering::noise::perlin::perlin_2d;

/// Computes the 2D curl magnitude of a scalar field
pub fn curl_image_2d(width: usize, height: usize, res: usize) -> Vec<u8> {
    let perlin: Vec<u8> = perlin_2d(width, height, res, 1.0);

    // Compute raw curl magnitudes
    let mut mags = Vec::with_capacity(width * height);
    let mut max_mag = 0.0f32;

    for y in 0..height {
        let y_plus = (y + 1) % height;
        let y_minus = (y + height - 1) % height;
        for x in 0..width {
            let x_plus = (x + 1) % width;
            let x_minus = (x + width - 1) % width;

            // Central differences, normalized to [0..1]
            let dy = (perlin[y_plus * width + x] as f32 - perlin[y_minus * width + x] as f32) * 0.5
                / 255.0;
            let dx = (perlin[y * width + x_plus] as f32 - perlin[y * width + x_minus] as f32) * 0.5
                / 255.0;

            let mag = (dy * dy + dx * dx).sqrt();
            if mag > max_mag {
                max_mag = mag;
            }
            mags.push(mag);
        }
    }

    // Normalize to [0..255]
    mags.into_iter()
        .map(|m| (m / max_mag * 255.0).round() as u8)
        .collect()
}
