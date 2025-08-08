use bevy::{
    asset::RenderAssetUsages,
    image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use image::RgbaImage;
use std::path::Path;

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
pub fn create_noise_image(
    size: usize,
    depth: usize,
    format: TextureFormat,
    data: Vec<u8>,
) -> Image {
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
