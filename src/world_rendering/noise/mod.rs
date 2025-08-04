mod perlin;
mod worley;

use std::path::Path;

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
    tasks::ComputeTaskPool,
};
use image::RgbaImage;

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct NoiseTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub turbulence: Handle<Image>,
}

pub fn setup_noise_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let base_texture = {
        let size = 128;

        // Generate base Perlin noise and Worley noise at increasing frequencies
        let noise_data = ComputeTaskPool::get().scope(|scope| {
            scope.spawn(async move { spread(&perlin::perlin_image_3d(size, size, size, 5, 8)) });
            scope.spawn(async move { spread(&worley::worley_octave_3d(size, size, size, 3)) });
            scope.spawn(async move { spread(&worley::worley_octave_3d(size, size, size, 6)) });
            scope.spawn(async move { spread(&worley::worley_octave_3d(size, size, size, 12)) });
            scope.spawn(async move { spread(&worley::worley_octave_3d(size, size, size, 24)) });
        });
        let perlin = &noise_data[0];
        let worley1 = &noise_data[1];
        let worley2 = &noise_data[2];
        let worley3 = &noise_data[3];
        let worley4 = &noise_data[4];

        // Generate Perlin-Worley noise
        let perlin_worley: Vec<f32> = perlin
            .iter()
            .zip(worley1)
            .map(|(&v1, &v2)| {
                let v1 = v1 as f32;
                let v2 = v2 as f32;
                let c = v1.clamp(v2, 1.0);
                map_range(c, 0.0, 1.0, v2, 1.0)
            })
            .collect();

        // Save the first depth layer of each noise map using image crate as a png image in assets/textures
        let save_noise_layer = |data: &Vec<f32>, filename: &str| {
            RgbaImage::from_raw(
                size as u32,
                size as u32,
                (&data[0..(size * size)])
                    .iter()
                    .flat_map(|&v| {
                        let val = (v * 255.0) as u8;
                        [val, val, val, 255]
                    })
                    .collect(),
            )
            .unwrap()
            .save(&Path::new("assets/textures").join(filename))
            .unwrap();
        };
        save_noise_layer(&perlin_worley, "perlinworley.png");
        save_noise_layer(worley2, "worley1.png");
        save_noise_layer(worley3, "worley2.png");
        save_noise_layer(worley4, "worley3.png");

        // Interleave the noise into RGBA floats
        let flat_data: Vec<f32> = (0..perlin.len())
            .flat_map(|i| [perlin_worley[i], worley2[i], worley3[i], worley4[i]])
            .collect();
        let data_u8 = bytemuck::cast_slice(&flat_data);
        let mut image = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: size as u32,
            },
            TextureDimension::D3,
            data_u8.to_vec(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..default()
        });
        image
    };

    let handle = images.add(base_texture);
    commands.insert_resource(NoiseTextures {
        base: handle.clone(),
        detail: handle.clone(),
        turbulence: handle,
    });
}

/// Stretch contrast: map [min, max] → [0,255].
fn spread(image: &[f32]) -> Vec<f32> {
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
#[inline]
fn map_range(val: f32, smin: f32, smax: f32, dmin: f32, dmax: f32) -> f32 {
    if smin == smax {
        return dmax;
    }
    (val - smin) / (smax - smin) * (dmax - dmin) + dmin
}
