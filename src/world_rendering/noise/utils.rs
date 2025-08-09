use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use image::RgbaImage;
use rayon::prelude::*;
use std::{fs, path::Path};

/// Generate a 3D noise texture using the provided noise function.
pub fn generate_noise_3d<F>(size: usize, depth: usize, noise_fn: F) -> Vec<f32>
where
    F: Fn(usize, Vec3A) -> f32 + Send + Sync,
{
    (0..(size * size * depth))
        .into_par_iter()
        .map(|i| {
            // Unravel i into x,y,z
            let x = i % size;
            let y = (i / size) % size;
            let z = i / (size * size);

            // Normalized coordinates in [0..1]
            let pos = Vec3A::new(
                x as f32 / size as f32,
                y as f32 / size as f32,
                z as f32 / depth as f32,
            );

            // Call the provided noise function to get v
            noise_fn(i, pos)
        })
        .collect()
}

// Generate FBM noise from a noise function
pub fn noise_fbm<F>(pos: Vec3A, octaves: usize, mut freq: f32, gain: f32, noise_fn: F) -> f32
where
    F: Fn(Vec3A, f32) -> f32 + Send + Sync,
{
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        total += noise_fn(pos * freq, freq) * amp;
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }
    total / norm
}

/// Stretch contrast: map [min, max] → [0,255].
pub fn spread(image: &[f32]) -> Vec<f32> {
    let mut minv: f32 = 1.0;
    let mut maxv: f32 = 0.0;
    for &v in image {
        minv = minv.min(v);
        maxv = maxv.max(v);
    }
    image
        .iter()
        .map(|&v| map_range(v, minv, maxv, 0.0, 1.0).clamp(0.0, 1.0))
        .collect()
}

/// Linearly map a value `x` in range [in_min..in_max] to [out_min..out_max].
pub fn map_range(val: f32, smin: f32, smax: f32, dmin: f32, dmax: f32) -> f32 {
    if (smax - smin).abs() < std::f32::EPSILON {
        return dmax;
    }
    (val - smin) / (smax - smin) * (dmax - dmin) + dmin
}

/// Converts a slice of f32 noise data into an RgbaImage and saves it.
pub fn save_noise_layer(data: &[f32], filename: &str, size: usize) {
    RgbaImage::from_raw(
        size as u32,
        size as u32,
        (&data[0..(size * size)])
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
    (0..noise_data[0].len())
        .flat_map(|i| {
            let mut channels = Vec::with_capacity(N);
            for j in 0..N {
                channels.push((noise_data[j][i] * 255.0).round() as u8);
            }
            channels
        })
        .collect()
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
    anisotropy_clamp: 1,
    border_color: None,
    label: None,
};

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
    let path = format!("assets/textures/{}.bin", path);
    let data = if let Ok(data) = fs::read(&path) {
        data
    } else {
        let data = generate_fn();
        save_texture_bin(&path, &data).unwrap();
        data
    };

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
