/// Stretch contrast: map [min, max] → [0,255].
pub fn spread(image: &[u8]) -> Vec<u8> {
    let mut minv: u8 = 255;
    let mut maxv: u8 = 0;
    for &v in image {
        minv = minv.min(v);
        maxv = maxv.max(v);
    }
    image
        .iter()
        .map(|&v| {
            map_range(v as f32, minv as f32, maxv as f32, 0.0, 255.0)
                .clamp(0.0, 255.0)
                .round() as u8
        })
        .collect()
}

/// Linearly map a value `x` in range [in_min..in_max] to [out_min..out_max].
pub fn map_range(val: f32, smin: f32, smax: f32, dmin: f32, dmax: f32) -> f32 {
    if (smax - smin).abs() < std::f32::EPSILON {
        return dmax;
    }
    (val - smin) / (smax - smin) * (dmax - dmin) + dmin
}
