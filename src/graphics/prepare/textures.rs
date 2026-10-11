use std::env;
use std::time::Instant;

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use rayon::prelude::*;
use skybound::scene::climate::weather_map;
use skybound::scene::noise::{fbm, lattice};

#[derive(Resource, ExtractResource, Clone)]
#[extract_app(RenderApp)]
pub struct VolumeTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub weather: Handle<Image>,
}

fn cellular(position: Vec3, period: i32, seed: u32) -> f32 {
    let p = position * period as f32;
    let cell = p.floor().as_ivec3();
    let mut nearest = 3.0f32;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let c = cell + ivec3(x, y, z);
                let feature = c.as_vec3()
                    + vec3(
                        lattice(c, period, seed),
                        lattice(c, period, seed.wrapping_add(31)),
                        lattice(c, period, seed.wrapping_add(67)),
                    );
                nearest = nearest.min(p.distance_squared(feature));
            }
        }
    }
    nearest.sqrt().min(1.0)
}

fn mip_chain<const N: usize>(mut level: Vec<u8>, mut size: UVec3) -> (Vec<u8>, u32) {
    let mut data = level.clone();
    let mut levels = 1;
    while size.max_element() > 1 {
        let next_size = (size / 2).max(UVec3::ONE);
        let mut next = vec![0; next_size.element_product() as usize * N];
        next.par_chunks_mut(N)
            .enumerate()
            .for_each(|(index, pixel)| {
                let index = index as u32;
                let origin = uvec3(
                    index % next_size.x,
                    (index / next_size.x) % next_size.y,
                    index / (next_size.x * next_size.y),
                ) * 2;
                let count = size.min(UVec3::splat(2));
                for (channel, byte) in pixel.iter_mut().enumerate() {
                    let mut sum = 0u32;
                    for z in 0..count.z {
                        for y in 0..count.y {
                            for x in 0..count.x {
                                let p = origin + uvec3(x, y, z);
                                sum += u32::from(
                                    level[((p.z * size.y + p.y) * size.x + p.x) as usize * N
                                        + channel],
                                );
                            }
                        }
                    }
                    *byte = ((sum + count.element_product() / 2) / count.element_product()) as u8;
                }
            });
        data.extend_from_slice(&next);
        level = next;
        size = next_size;
        levels += 1;
    }
    (data, levels)
}

fn texture<const N: usize>(
    size: u32,
    depth: u32,
    format: TextureFormat,
    sample: impl Fn(Vec3) -> [f32; N] + Sync,
) -> Image {
    let mut data = vec![0; (size * size * depth) as usize * N];
    data.par_chunks_mut(N).enumerate().for_each(|(i, pixel)| {
        let i = i as u32;
        let position = vec3(
            (i % size) as f32 / size as f32,
            ((i / size) % size) as f32 / size as f32,
            (i / (size * size)) as f32 / depth as f32,
        );
        for (byte, value) in pixel.iter_mut().zip(sample(position)) {
            *byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    });
    let (data, levels) = mip_chain::<N>(data, uvec3(size, size, depth));
    let mut image = Image::new_uninit(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: depth,
        },
        if depth == 1 {
            TextureDimension::D2
        } else {
            TextureDimension::D3
        },
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
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
}

pub fn prepare_volume_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let start = Instant::now();
    let mut arguments = env::args();
    let seed = arguments
        .find(|arg| arg == "--seed")
        .and_then(|_| arguments.next())
        .map_or(0, |value| {
            value
                .parse::<u32>()
                .expect("seed must be an unsigned integer")
        });
    let (base, (detail, weather)) = rayon::join(
        || {
            texture(128, 128, TextureFormat::Rgba8Unorm, |p| {
                let billow = 1.0 - (1.0 - fbm(p, 4, seed ^ 0x11)) * cellular(p, 4, seed ^ 0xd3);
                [
                    billow,
                    fbm(p, 4, seed ^ 0x1d),
                    fbm(p, 4, seed ^ 0x35),
                    fbm(p, 4, seed ^ 0x61),
                ]
            })
        },
        || {
            rayon::join(
                || {
                    texture(64, 64, TextureFormat::Rgba8Unorm, |p| {
                        [
                            cellular(p, 3, seed ^ 0x17),
                            cellular(p, 6, seed ^ 0x2b),
                            cellular(p, 12, seed ^ 0x47),
                            fbm(p, 4, seed ^ 0x71),
                        ]
                    })
                },
                || {
                    texture(256, 1, TextureFormat::Rgba8Unorm, |p| {
                        weather_map(p.truncate()).to_array()
                    })
                },
            )
        },
    );
    info!(
        "Prepared filtered procedural cloud and flow textures in {:?}",
        start.elapsed()
    );
    commands.insert_resource(VolumeTextures {
        base: images.add(base),
        detail: images.add(detail),
        weather: images.add(weather),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn volume_mips_average_all_axes_and_preserve_constant_fields() {
        let (data, levels) = mip_chain::<1>(vec![128; 64], UVec3::splat(4));
        assert_eq!(levels, 3);
        assert_eq!(data.len(), 64 + 8 + 1);
        assert!(data.iter().all(|value| *value == 128));
        let (data, levels) =
            mip_chain::<1>((0..8).map(|index| index * 32).collect(), UVec3::splat(2));
        assert_eq!(levels, 2);
        assert_eq!(data[8], 112);
        let (data, levels) = mip_chain::<1>(vec![64; 16], uvec3(4, 4, 1));
        assert_eq!(levels, 3);
        assert_eq!(data.len(), 16 + 4 + 1);
        assert!(data.iter().all(|value| *value == 64));
    }

    #[test]
    fn periodic_noise_tiles_with_matching_slopes() {
        let p = vec3(0.0, 0.23, 0.61);
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!(
                (fbm(p, 4, 17) - fbm(p + axis, 4, 17)).abs() < 1e-5,
                "noise seam"
            );
            let left = fbm(p - axis * 0.0001, 4, 17);
            let right = fbm(p + axis * 0.0001, 4, 17);
            assert!((left - right).abs() < 0.005, "noise derivative seam");
        }
    }
}
