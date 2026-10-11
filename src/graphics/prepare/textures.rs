use std::env;
use std::time::Instant;

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use rayon::prelude::*;

#[derive(Resource, ExtractResource, Clone)]
#[extract_app(RenderApp)]
pub struct VolumeTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub weather: Handle<Image>,
}

fn lattice(cell: IVec3, period: i32, seed: u32) -> f32 {
    let mut value = (cell.x.rem_euclid(period) as u32).wrapping_mul(73_856_093)
        ^ (cell.y.rem_euclid(period) as u32).wrapping_mul(19_349_663)
        ^ (cell.z.rem_euclid(period) as u32).wrapping_mul(83_492_791)
        ^ seed;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value >> 8) as f32 / 16_777_215.0
}

fn periodic_noise(position: Vec3, period: i32, seed: u32) -> f32 {
    let position = position * period as f32;
    let cell = position.floor().as_ivec3();
    let fraction = position - position.floor();
    let weight = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);
    let mut value = 0.0;
    for corner in 0..8 {
        let offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let blend = Vec3::select(offset.cmpeq(IVec3::ZERO), Vec3::ONE - weight, weight);
        let hash = (lattice(cell + offset, period, seed) * 255.0) as u32 & 15;
        let delta = fraction - offset.as_vec3();
        let u = if hash < 8 { delta.x } else { delta.y };
        let v = if hash < 4 {
            delta.y
        } else if hash == 12 || hash == 14 {
            delta.x
        } else {
            delta.z
        };
        let gradient = (if hash & 1 == 0 { u } else { -u }) + (if hash & 2 == 0 { v } else { -v });
        value += gradient * blend.x * blend.y * blend.z;
    }
    value * 0.5 + 0.5
}

fn fbm(position: Vec3, period: i32, seed: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    for octave in 0..4 {
        value += periodic_noise(position, period << octave, seed.wrapping_add(octave * 1013))
            * amplitude;
        amplitude *= 0.5;
    }
    ((value / 1.875 - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0)
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
    let mut image = Image::new(
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
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
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
                        [
                            fbm(p, 4, seed ^ 0x83),
                            fbm(p, 4, seed ^ 0xc5),
                            fbm(p, 2, seed ^ 0xf1),
                            fbm(p, 4, seed ^ 0x133),
                        ]
                    })
                },
            )
        },
    );
    info!(
        "Prepared 9.25 MiB of procedural cloud and flow textures in {:?}",
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
