mod curl;
mod perlin;
mod simplex;
mod utils;
mod worley;

use crate::world_rendering::noise::{
    curl::curl_2d_texture,
    perlin::perlin_3d,
    simplex::simplex_3d,
    utils::{interleave_channels, load_or_generate_texture, map_range, save_noise_layer, spread},
    worley::worley_3d,
};
use bevy::{
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::TextureFormat},
};

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct NoiseTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
    pub turbulence: Handle<Image>,
    pub weather: Handle<Image>,
    pub fog: Handle<Image>,
}

pub fn setup_noise_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let start = std::time::Instant::now();

    let size = 192;
    let depth = 64;
    let base_texture = load_or_generate_texture(
        "base_texture",
        size,
        depth,
        TextureFormat::Rgba8Unorm,
        || {
            let perlinworley = spread(&perlin_3d(size, depth, 5, 0.5, Vec2::new(16.0, 8.0), 1.5))
                .iter()
                .zip(worley_3d(size, depth, 16.0, 2.0, true))
                .map(|(&perlin, worley)| map_range(perlin, 0.0, 1.0, worley, 1.0))
                .collect::<Vec<f32>>();
            let gamma = 0.5;
            let worley1 = spread(&worley_3d(size, depth, 12.0, gamma, true));
            let worley2 = spread(&worley_3d(size, depth, 18.0, gamma, true));
            let worley3 = spread(&worley_3d(size, depth, 24.0, gamma, true));

            save_noise_layer(&perlinworley, "perlinworley.png", size);
            save_noise_layer(&worley1, "worley1.png", size);
            save_noise_layer(&worley2, "worley2.png", size);
            save_noise_layer(&worley3, "worley3.png", size);

            interleave_channels([perlinworley, worley1, worley2, worley3])
        },
    );

    let size = 64;
    let detail_texture = load_or_generate_texture(
        "detail_texture",
        size,
        size,
        TextureFormat::Rgba8Unorm,
        || {
            let gamma = 0.6;
            let detail1 = worley_3d(size, size, 5.0, gamma, true);
            let detail2 = worley_3d(size, size, 6.0, gamma, true);
            let detail3 = worley_3d(size, size, 7.0, gamma, true);

            save_noise_layer(&detail1, "detail1.png", size);
            save_noise_layer(&detail2, "detail2.png", size);
            save_noise_layer(&detail3, "detail3.png", size);

            interleave_channels([detail1, detail2, detail3, vec![0.0; size * size * size]])
        },
    );

    let size = 512;
    let turbulence_texture = load_or_generate_texture(
        "turbulence_texture",
        size,
        1,
        TextureFormat::Rgba8Unorm,
        || {
            let (curl1, curl2, curl3) = curl_2d_texture(size, size);

            save_noise_layer(&curl1, "curl1.png", size);
            save_noise_layer(&curl2, "curl2.png", size);
            save_noise_layer(&curl3, "curl3.png", size);

            interleave_channels([curl1, curl2, curl3, vec![0.0; size * size * size]])
        },
    );

    let size = 256;
    let depth = 12;
    let weather_texture = load_or_generate_texture(
        "weather_texture",
        size,
        depth,
        TextureFormat::Rgba8Unorm,
        || {
            let weather1 = spread(&perlin_3d(size, depth, 5, 0.45, Vec2::new(5.0, 2.0), 2.0))
                .iter()
                .map(|&x| x * 1.1 - 0.1)
                .collect::<Vec<f32>>();
            let weather2 = spread(&worley_3d(size, depth, 8.0, 0.4, true))
                .iter()
                .zip(spread(&perlin_3d(size, depth, 5, 0.8, Vec2::new(8.0, 2.0), 1.0)).iter())
                .map(|(&a, &b)| (a * 1.7 - 0.5) + b * 0.1)
                .collect::<Vec<f32>>();
            let weather3 = spread(&perlin_3d(size, depth, 1, 0.8, Vec2::new(4.0, 1.0), 0.5))
                .iter()
                .zip(spread(&worley_3d(size, depth, 4.0, 1.0, false)).iter())
                .map(|(&a, &b)| (a * 1.2 - 0.2) - b * 0.1)
                .collect::<Vec<f32>>();

            save_noise_layer(&weather1, "weather1.png", size);
            save_noise_layer(&weather2, "weather2.png", size);
            save_noise_layer(&weather3, "weather3.png", size);

            interleave_channels([weather1, weather2, weather3, vec![0.0; size * size * size]])
        },
    );

    let size = 96;
    let fog_texture =
        load_or_generate_texture("fog_texture", size, size, TextureFormat::Rg8Unorm, || {
            let fog1 = spread(&perlin_3d(size, size, 6, 0.1, Vec2::new(18.0, 12.0), 1.0));
            let fog2 = spread(&simplex_3d(size, size, 12, 0.4, 6.0, 1.0));

            save_noise_layer(&fog1, "fog1.png", size);
            save_noise_layer(&fog2, "fog2.png", size);

            interleave_channels([fog1, fog2])
        });

    println!("Noise generation took: {:?}", start.elapsed());
    commands.insert_resource(NoiseTextures {
        base: images.add(base_texture),
        detail: images.add(detail_texture),
        turbulence: images.add(turbulence_texture),
        weather: images.add(weather_texture),
        fog: images.add(fog_texture),
    });
}
