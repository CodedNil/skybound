mod simplex;
mod utils;
mod worley;

use crate::world_rendering::noise::{
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
    pub weather: Handle<Image>,
}

pub fn setup_noise_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let start = std::time::Instant::now();

    let size = 384;
    let depth = 96;
    let base_texture =
        load_or_generate_texture("base_texture", size, depth, TextureFormat::R8Unorm, || {
            let perlinworley = spread(&simplex_3d(size, depth, 5, 0.5, Vec2::new(8.0, 6.0), 1.5))
                .iter()
                .zip(worley_3d(size, depth, 3, 0.5, 12.0, 0.5))
                .map(|(&perlin, worley)| map_range(perlin, -(1.0 - worley), 1.0, 0.0, 1.0))
                .collect::<Vec<f32>>();

            save_noise_layer(&perlinworley, "perlinworley.png", size);
            interleave_channels([perlinworley])
        });

    let size = 192;
    let depth = 96;
    let detail_texture = load_or_generate_texture(
        "detail_texture",
        size,
        depth,
        TextureFormat::Rgba8Unorm,
        || {
            let detail1 = worley_3d(size, depth, 6, 0.8, 8.0, 0.6);
            let fog1 = spread(&simplex_3d(size, depth, 4, 0.2, Vec2::new(7.0, 7.0), 1.0));
            let fog2 = spread(&simplex_3d(size, depth, 6, 0.4, Vec2::new(6.0, 4.0), 1.0));

            save_noise_layer(&detail1, "detail1.png", size);
            save_noise_layer(&fog1, "fog1.png", size);
            save_noise_layer(&fog2, "fog2.png", size);
            interleave_channels([detail1, fog1, fog2, vec![0.0; size * size * size]])
        },
    );

    let size = 512;
    let depth = 1;
    let weather_texture = load_or_generate_texture(
        "weather_texture",
        size,
        depth,
        TextureFormat::Rgba8Unorm,
        || {
            let weather1 = spread(&simplex_3d(size, depth, 5, 0.45, Vec2::new(5.0, 2.0), 2.0))
                .iter()
                .map(|&x| x * 1.1 - 0.1)
                .collect::<Vec<f32>>();
            let weather2 = spread(&worley_3d(size, depth, 3, 0.5, 8.0, 0.4))
                .iter()
                .zip(spread(&simplex_3d(size, depth, 5, 0.8, Vec2::new(8.0, 2.0), 1.0)).iter())
                .map(|(&a, &b)| (a * 1.7 - 0.5) + b * 0.1)
                .collect::<Vec<f32>>();
            let weather3 = spread(&simplex_3d(size, depth, 6, 0.8, Vec2::splat(2.0), 0.5))
                .iter()
                .zip(spread(&worley_3d(size, depth, 3, 0.5, 2.0, 1.0)).iter())
                .map(|(&a, &b)| (a * 1.2 - 0.2) - b * 0.1)
                .collect::<Vec<f32>>();
            let height = spread(&simplex_3d(size, depth, 6, 0.8, Vec2::splat(8.0), 0.5))
                .iter()
                .zip(spread(&worley_3d(size, depth, 3, 0.5, 7.0, 1.0)).iter())
                .map(|(&a, &b)| (a * 1.2 - 0.2) - b * 0.1)
                .collect::<Vec<f32>>();

            save_noise_layer(&weather1, "weather1.png", size);
            save_noise_layer(&weather2, "weather2.png", size);
            save_noise_layer(&weather3, "weather3.png", size);
            save_noise_layer(&height, "height.png", size);
            interleave_channels([weather1, weather2, weather3, height])
        },
    );

    println!("Noise generation took: {:?}", start.elapsed());
    commands.insert_resource(NoiseTextures {
        base: images.add(base_texture),
        detail: images.add(detail_texture),
        weather: images.add(weather_texture),
    });
}
