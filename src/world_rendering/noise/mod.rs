mod curl;
mod perlin;
mod simplex;
mod worley;

use crate::world_rendering::noise::{
    curl::curl_2d_texture, perlin::perlin_3d, simplex::simplex_3d, worley::worley_3d,
};
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};
use image::RgbaImage;
use std::path::Path;

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct NoiseTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub turbulence: Handle<Image>,
    pub fog: Handle<Image>,
}

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

pub fn setup_noise_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let start = std::time::Instant::now();

    // Save the first depth layer of each noise map using image crate as a png image in assets/textures
    let save_noise_layer = |data: &[u8], filename: &str, size: usize| {
        RgbaImage::from_raw(
            size as u32,
            size as u32,
            (&data[0..(size * size)])
                .iter()
                .flat_map(|&val| [val, val, val, 255])
                .collect(),
        )
        .unwrap()
        .save(&Path::new("assets/textures").join(filename))
        .unwrap();
    };

    let base_texture = {
        let size = 128;

        // Generate base Perlin noise and Worley noise at increasing frequencies
        let worley_pow = 0.5;
        let perlin = spread(&perlin_3d(size, size, size, 5, 0.55, 7.0, 0.7));
        let worley1 = worley_3d(size, size, size, 6, worley_pow);
        let worley2 = spread(&worley_3d(size, size, size, 12, worley_pow));
        let worley3 = spread(&worley_3d(size, size, size, 18, worley_pow));
        let worley4 = spread(&worley_3d(size, size, size, 24, worley_pow));

        // Generate Perlin-Worley noise
        let perlin_worley: Vec<u8> = perlin
            .iter()
            .zip(worley1)
            .map(|(&perlin, worley1)| {
                let perlin = perlin as f32 / 255.0;
                let worley1 = worley1 as f32 / 255.0;
                (map_range(perlin.abs() * 2.0 - 1.0, 0.0, 1.0, worley1, 1.0) * 255.0).round() as u8
            })
            .collect();

        save_noise_layer(&perlin, "perlinworley.png", size);
        save_noise_layer(&worley2, "worley1.png", size);
        save_noise_layer(&worley3, "worley2.png", size);
        save_noise_layer(&worley4, "worley3.png", size);

        // Interleave the noise into RGBA floats
        let flat_data: Vec<u8> = (0..perlin.len())
            .flat_map(|i| [perlin_worley[i], worley2[i], worley3[i], worley4[i]])
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: size as u32,
            },
            TextureDimension::D3,
            bytemuck::cast_slice(&flat_data).to_vec(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(IMAGE_SAMPLER);
        image
    };

    let detail_texture = {
        let size = 32;

        // Generate Worley noise at increasing frequencies
        let worley_pow = 0.6;
        let worley1 = worley_3d(size, size, size, 5, worley_pow);
        let worley2 = worley_3d(size, size, size, 6, worley_pow);
        let worley3 = worley_3d(size, size, size, 7, worley_pow);

        save_noise_layer(&worley1, "detail1.png", size);
        save_noise_layer(&worley2, "detail2.png", size);
        save_noise_layer(&worley3, "detail3.png", size);

        // Interleave the noise into RGBA floats
        let flat_data: Vec<u8> = (0..worley1.len())
            .flat_map(|i| [worley1[i], worley2[i], worley3[i], 255])
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: size as u32,
            },
            TextureDimension::D3,
            bytemuck::cast_slice(&flat_data).to_vec(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(IMAGE_SAMPLER);
        image
    };

    let turbulence_texture = {
        let size = 128;

        // Generate Curl noise at increasing octaves
        let (curl1, curl2, curl3) = curl_2d_texture(size, size);

        save_noise_layer(&curl1, "curl1.png", size);
        save_noise_layer(&curl2, "curl2.png", size);
        save_noise_layer(&curl3, "curl3.png", size);

        // Interleave the noise into RGBA floats
        let flat_data: Vec<u8> = (0..curl1.len())
            .flat_map(|i| [curl1[i], curl2[i], curl3[i], 255])
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&flat_data).to_vec(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(IMAGE_SAMPLER);
        image
    };

    let fog_texture = {
        let size = 128;

        // Generate Simplex noise at increasing octaves
        let fog1 = spread(&simplex_3d(size, size, size, 6, 0.1, 18.0, 1.0, true)); // The fogs heightmap
        let fog2 = spread(&simplex_3d(size, size, size, 12, 0.4, 6.0, 1.0, false)); // The fine noise for the fog
        let fog3 = spread(&simplex_3d(size, size, size, 5, 0.5, 6.0, 1.0, false)); // The fogs color pattern

        save_noise_layer(&fog1, "fog1.png", size);
        save_noise_layer(&fog2, "fog2.png", size);
        save_noise_layer(&fog3, "fog3.png", size);

        // Interleave the noise into RGBA floats
        let flat_data: Vec<u8> = (0..fog1.len())
            .flat_map(|i| [fog1[i], fog2[i], fog3[i], 255])
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: size as u32,
            },
            TextureDimension::D3,
            bytemuck::cast_slice(&flat_data).to_vec(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(IMAGE_SAMPLER);
        image
    };

    println!("Noise generation took: {:?}", start.elapsed());
    commands.insert_resource(NoiseTextures {
        base: images.add(base_texture),
        detail: images.add(detail_texture),
        turbulence: images.add(turbulence_texture),
        fog: images.add(fog_texture),
    });
}

/// Stretch contrast: map [min, max] → [0,255].
fn spread(image: &[u8]) -> Vec<u8> {
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
#[inline]
fn map_range(val: f32, smin: f32, smax: f32, dmin: f32, dmax: f32) -> f32 {
    if (smax - smin).abs() < std::f32::EPSILON {
        return dmax;
    }
    (val - smin) / (smax - smin) * (dmax - dmin) + dmin
}
