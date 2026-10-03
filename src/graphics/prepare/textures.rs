use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};
use rayon::prelude::*;
use std::time::Instant;

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
        value += lattice(cell + offset, period, seed) * blend.x * blend.y * blend.z;
    }
    value
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
    let (base, (detail, weather)) = rayon::join(
        || texture(64, 64, TextureFormat::R8Unorm, |p| [fbm(p, 4, 17)]),
        || {
            rayon::join(
                || {
                    texture(64, 64, TextureFormat::Rgba8Unorm, |p| {
                        [fbm(p, 4, 29), fbm(p, 2, 53), fbm(p, 2, 97), 0.0]
                    })
                },
                || {
                    texture(128, 1, TextureFormat::Rg8Unorm, |p| {
                        [fbm(p, 4, 131), fbm(p, 4, 197)]
                    })
                },
            )
        },
    );
    info!(
        "Prepared 1.28 MiB of procedural textures in {:?}",
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
    #[test]
    fn textures_pack_channels_and_have_useful_variation() {
        let image = texture(8, 1, TextureFormat::Rg8Unorm, |p| {
            [fbm(p, 2, 131), fbm(p, 2, 197)]
        });
        let data = image.data.expect("generated pixels");
        assert_eq!(data.len(), 8 * 8 * 2, "incorrect channel packing");
        let (min, max) = data
            .iter()
            .fold((255, 0), |(min, max), &v| (min.min(v), max.max(v)));
        assert!(max - min > 100, "noise lost contrast");
        assert!(
            data.as_chunks::<2>()
                .0
                .iter()
                .any(|pixel| pixel[0] != pixel[1]),
            "noise channels are correlated"
        );
    }
}
