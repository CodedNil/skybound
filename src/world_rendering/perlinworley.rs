use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct PerlinWorleyTextureHandle {
    pub handle: Handle<Image>,
}

pub fn setup_perlinworley_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = 128;
    let mut data = vec![0f32; size * size * size * 4]; // RGBA

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let pos = Vec3::new(x as f32, y as f32, z as f32) / size as f32;

                let mut pfbm = mix(1.0, perlin_fbm(pos, 4.0, 7), 0.5);
                pfbm = (pfbm * 2.0 - 1.0).abs(); // Billowy perlin noise [0,1]

                let freq = 4.0;
                let worley1 = worley_fbm(pos, freq);
                let worley2 = worley_fbm(pos, freq * 2.0);
                let worley3 = worley_fbm(pos, freq * 4.0);
                let perlin_worley = remap(pfbm, 0.0, 1.0, worley1, 1.0);

                let idx = (z * size * size + y * size + x) * 4;
                data[idx] = perlin_worley; // R
                data[idx + 1] = worley1; // G
                data[idx + 2] = worley2; // B
                data[idx + 3] = worley3; // A
            }
        }
    }

    let data_u8 = bytemuck::cast_slice(&data);
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

    let handle = images.add(image);
    commands.insert_resource(PerlinWorleyTextureHandle { handle });
}

// Linear interpolation between two values
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// Remap a range
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    (((x - a) / (b - a)) * (d - c)) + c
}

// White noise integer hash: vec2 → vec2 [-1,1]
const UI3: UVec3 = UVec3::new(1597334673, 3812015801, 2798796415);
const UIF: f32 = 1.0 / 4294967295.0;
fn hash33(p: Vec3) -> Vec3 {
    let q = p.as_ivec3().as_uvec3().wrapping_mul(UI3);
    let hash = UVec3::splat(q.x ^ q.y ^ q.z).wrapping_mul(UI3);
    -1.0 + 2.0 * Vec3::new(hash.x as f32, hash.y as f32, hash.z as f32) * UIF
}

// Perlin tiled noise: vec3 → f32 [-1,1]
fn perlin_noise(x: Vec3, freq: f32) -> f32 {
    let p = x.floor();
    let w = x.fract();

    let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    let ga = hash33(p % freq);
    let gb = hash33((p + Vec3::new(1.0, 0.0, 0.0)) % freq);
    let gc = hash33((p + Vec3::new(0.0, 1.0, 0.0)) % freq);
    let gd = hash33((p + Vec3::new(1.0, 1.0, 0.0)) % freq);
    let ge = hash33((p + Vec3::new(0.0, 0.0, 1.0)) % freq);
    let gf = hash33((p + Vec3::new(1.0, 0.0, 1.0)) % freq);
    let gg = hash33((p + Vec3::new(0.0, 1.0, 1.0)) % freq);
    let gh = hash33((p + Vec3::new(1.0, 1.0, 1.0)) % freq);

    let va = ga.dot(w - Vec3::new(0.0, 0.0, 0.0));
    let vb = gb.dot(w - Vec3::new(1.0, 0.0, 0.0));
    let vc = gc.dot(w - Vec3::new(0.0, 1.0, 0.0));
    let vd = gd.dot(w - Vec3::new(1.0, 1.0, 0.0));
    let ve = ge.dot(w - Vec3::new(0.0, 0.0, 1.0));
    let vf = gf.dot(w - Vec3::new(1.0, 0.0, 1.0));
    let vg = gg.dot(w - Vec3::new(0.0, 1.0, 1.0));
    let vh = gh.dot(w - Vec3::new(1.0, 1.0, 1.0));

    va + u.x * (vb - va)
        + u.y * (vc - va)
        + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd)
        + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf)
        + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh)
}

// Cellular tiled noise: vec3 → f32 [0,1]
fn worley_noise(uv: Vec3, freq: f32) -> f32 {
    let cell = uv.floor();
    let fractal = uv.fract();

    let mut min_dist = 10000.0f32;
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let offset = Vec3::new(x as f32, y as f32, z as f32);
                let point = hash33((cell + offset) % freq) * 0.5 + 0.5 + offset;
                let dist = fractal - point;
                min_dist = min_dist.min(dist.dot(dist));
            }
        }
    }

    1.0 - min_dist
}

// Tileable Perlin-FBM noise [-1,1]
fn perlin_fbm(p: Vec3, freq: f32, octaves: u32) -> f32 {
    let g = -0.85f32.powf(2.0);
    let mut amp = 1.0;
    let mut noise = 0.0;
    let mut freq = freq;
    for _ in 0..octaves {
        noise += amp * perlin_noise(p * freq, freq);
        freq *= 2.0;
        amp *= g;
    }
    return noise;
}

// Tileable Worley-FBM noise [0,1]
fn worley_fbm(p: Vec3, freq: f32) -> f32 {
    return worley_noise(p * freq, freq) * 0.625
        + worley_noise(p * freq * 2.0, freq * 2.0) * 0.25
        + worley_noise(p * freq * 4.0, freq * 4.0) * 0.125;
}
