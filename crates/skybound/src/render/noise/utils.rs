use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use image::RgbaImage;
use std::{fs, path::Path};

/// Stretch contrast: map [min, max] → [0,255].
pub fn spread(image: &[f32]) -> Vec<f32> {
    let (minv, maxv) = image
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    image
        .iter()
        .map(|&v| map_range(v, minv, maxv, 0.0, 1.0).clamp(0.0, 1.0))
        .collect()
}

/// Linearly map a value `x` in range [`in_min..in_max`] to [`out_min..out_max`].
pub fn map_range(val: f32, smin: f32, smax: f32, dmin: f32, dmax: f32) -> f32 {
    if (smax - smin).abs() < f32::EPSILON {
        return dmax;
    }
    (val - smin) / (smax - smin) * (dmax - dmin) + dmin
}

/// Converts a slice of f32 noise data into an `RgbaImage` and saves it.
pub fn save_noise_layer(data: &[f32], filename: &str, size: usize) {
    RgbaImage::from_raw(
        size as u32,
        size as u32,
        data[0..(size * size)]
            .iter()
            .flat_map(|&val| {
                let val = (val * 255.0).round() as u8;
                [val, val, val, 255]
            })
            .collect(),
    )
    .unwrap()
    .save(&Path::new("assets/textures").join(filename))
    .unwrap();
}

/// Helper to interleave multiple noise vectors into a combined `Vec<u8>`.
pub fn interleave_channels<const N: usize>(noise_data: [Vec<f32>; N]) -> Vec<u8> {
    let len = noise_data[0].len();
    let mut out = Vec::with_capacity(len * N);
    for i in 0..len {
        for ch in &noise_data {
            out.push((ch[i] * 255.0).round() as u8);
        }
    }
    out
}

/// Creates a new Bevy `Image` from noise data.
const IMAGE_SAMPLER: ImageSamplerDescriptor = ImageSamplerDescriptor {
    address_mode_u: ImageAddressMode::Repeat,
    address_mode_v: ImageAddressMode::Repeat,
    address_mode_w: ImageAddressMode::Repeat,
    mag_filter: ImageFilterMode::Linear,
    min_filter: ImageFilterMode::Linear,
    mipmap_filter: ImageFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: 4,
    border_color: None,
    label: None,
};

/// Write raw texture bytes to a file path.
pub fn save_texture_bin(path: &str, data: &[u8]) -> std::io::Result<()> {
    fs::write(path, data)
}

pub fn load_or_generate_texture<F>(
    path: &str,
    size: usize,
    depth: usize,
    format: TextureFormat,
    generate_fn: F,
) -> Image
where
    F: FnOnce() -> Vec<u8>,
{
    let path = format!("assets/textures/{path}.bin");
    let data = fs::read(&path).unwrap_or_else(|_| {
        let data = generate_fn();
        save_texture_bin(&path, &data).unwrap();
        data
    });

    let mut image = Image::new(
        Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: depth as u32,
        },
        if depth > 1 {
            TextureDimension::D3
        } else {
            TextureDimension::D2
        },
        bytemuck::cast_slice(&data).to_vec(),
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(IMAGE_SAMPLER);
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_range_and_spread() {
        let v = map_range(5.0, 0.0, 10.0, 0.0, 1.0);
        assert!((v - 0.5).abs() < 1e-6);

        let img = vec![0.2f32, 0.8f32];
        let out = spread(&img);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_interleave_channels() {
        let a = vec![0.0f32, 0.5, 1.0];
        let b = vec![1.0f32, 0.5, 0.0];
        let out = interleave_channels([a, b]);
        assert_eq!(out.len(), 6);
        // first pixel from channel a then b
        assert_eq!(out[0], (0.0f32 * 255.0).round() as u8);
        assert_eq!(out[1], (1.0f32 * 255.0).round() as u8);
    }
}
