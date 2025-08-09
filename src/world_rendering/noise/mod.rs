mod curl;
mod perlin;
mod simplex;
mod utils;
mod worley;

use crate::world_rendering::noise::{
    curl::curl_2d_texture,
    perlin::{perlin_3d, perlin3},
    simplex::simplex_3d,
    utils::{
        create_noise_image, generate_noise_3d, interleave_channels, map_range, noise_fbm,
        save_noise_layer, spread,
    },
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

    let base_texture = {
        let start = std::time::Instant::now();
        let size = 128;

        let worleya = worley_3d(size, size, 6.0, 0.5, true);
        let perlinworley = spread(&generate_noise_3d(size, size, |i, pos| {
            let perlin = noise_fbm(pos, 5, 0.5, 7.0, |pos, freq| perlin3(pos, freq, true));
            let perlin = (perlin * 0.5 + 0.5).powf(0.7);
            map_range(perlin.abs() * 2.0 - 1.0, 0.0, 1.0, worleya[i], 1.0)
        }));
        let gamma = 0.5;
        let worley1 = spread(&worley_3d(size, size, 12.0, gamma, true));
        let worley2 = spread(&worley_3d(size, size, 18.0, gamma, true));
        let worley3 = spread(&worley_3d(size, size, 24.0, gamma, true));

        save_noise_layer(&perlinworley, "perlinworley.png", size);
        save_noise_layer(&worley1, "worley1.png", size);
        save_noise_layer(&worley2, "worley2.png", size);
        save_noise_layer(&worley3, "worley3.png", size);

        let flat_data = interleave_channels([perlinworley, worley1, worley2, worley3]);
        let image = create_noise_image(size, size, TextureFormat::Rgba8Unorm, flat_data);
        println!("Base texture generation took: {:?}", start.elapsed());
        image
    };

    let detail_texture = {
        let start = std::time::Instant::now();
        let size = 32;

        let gamma = 0.6;
        let detail1 = worley_3d(size, size, 5.0, gamma, true);
        let detail2 = worley_3d(size, size, 6.0, gamma, true);
        let detail3 = worley_3d(size, size, 7.0, gamma, true);

        save_noise_layer(&detail1, "detail1.png", size);
        save_noise_layer(&detail2, "detail2.png", size);
        save_noise_layer(&detail3, "detail3.png", size);

        let flat_data =
            interleave_channels([detail1, detail2, detail3, vec![0.0; size * size * size]]);
        let image = create_noise_image(size, size, TextureFormat::Rgba8Unorm, flat_data);
        println!("Detail texture generation took: {:?}", start.elapsed());
        image
    };

    let turbulence_texture = {
        let start = std::time::Instant::now();
        let size = 128;

        let (curl1, curl2, curl3) = curl_2d_texture(size, size);

        save_noise_layer(&curl1, "curl1.png", size);
        save_noise_layer(&curl2, "curl2.png", size);
        save_noise_layer(&curl3, "curl3.png", size);

        let flat_data = interleave_channels([curl1, curl2, curl3, vec![0.0; size * size * size]]);
        let image = create_noise_image(size, 1, TextureFormat::Rgba8Unorm, flat_data);
        println!("Turbulence texture generation took: {:?}", start.elapsed());
        image
    };

    let weather_texture = {
        let start = std::time::Instant::now();
        let size = 512;

        let weather1 = spread(&perlin_3d(size, 1, 5, 0.45, 5.0, 2.0))
            .iter()
            .map(|&x| x * 1.1 - 0.1)
            .collect::<Vec<f32>>();
        let weather2 = spread(&worley_3d(size, 1, 8.0, 0.4, true))
            .iter()
            .zip(spread(&perlin_3d(size, 1, 5, 0.8, 8.0, 1.0)).iter())
            .map(|(&a, &b)| (a * 1.7 - 0.5) + b * 0.1)
            .collect::<Vec<f32>>();
        let weather3 = spread(&perlin_3d(size, 1, 1, 0.8, 4.0, 0.5))
            .iter()
            .zip(spread(&worley_3d(size, 1, 4.0, 1.0, false)).iter())
            .map(|(&a, &b)| (a * 1.2 - 0.2) - b * 0.1)
            .collect::<Vec<f32>>();

        save_noise_layer(&weather1, "weather1.png", size);
        save_noise_layer(&weather2, "weather2.png", size);
        save_noise_layer(&weather3, "weather3.png", size);

        let flat_data =
            interleave_channels([weather1, weather2, weather3, vec![0.0; size * size * size]]);
        let image = create_noise_image(size, 1, TextureFormat::Rgba8Unorm, flat_data);
        println!("Weather texture generation took: {:?}", start.elapsed());
        image
    };

    let fog_texture = {
        let start = std::time::Instant::now();
        let size = 128;

        let fog1 = spread(&perlin_3d(size, size, 6, 0.1, 18.0, 1.0));
        let fog2 = spread(&simplex_3d(size, size, 12, 0.4, 6.0, 1.0));

        save_noise_layer(&fog1, "fog1.png", size);
        save_noise_layer(&fog2, "fog2.png", size);

        let flat_data = interleave_channels([fog1, fog2]);
        let image = create_noise_image(size, size, TextureFormat::Rg8Unorm, flat_data);
        println!("Fog texture generation took: {:?}", start.elapsed());
        image
    };

    println!("Noise generation took: {:?}", start.elapsed());
    commands.insert_resource(NoiseTextures {
        base: images.add(base_texture),
        detail: images.add(detail_texture),
        turbulence: images.add(turbulence_texture),
        weather: images.add(weather_texture),
        fog: images.add(fog_texture),
    });
}
