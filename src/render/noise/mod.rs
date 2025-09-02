mod simplex;
mod utils;
mod worley;

use bevy::{
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::TextureFormat},
};
use simplex::simplex_3d;
use utils::{interleave_channels, load_or_generate_texture, map_range, save_noise_layer, spread};
use worley::worley_3d;

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct NoiseTextures {
    pub base: Handle<Image>,
    pub detail: Handle<Image>,
}

/// Generates or loads procedural noise textures and inserts them as resources.
pub fn setup_noise_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let start = std::time::Instant::now();

    let size = 384;
    let depth = 96;
    let base_texture =
        load_or_generate_texture("base_texture", size, depth, TextureFormat::R8Unorm, || {
            let perlinworley = spread(
                &simplex_3d(size, depth, 7, 0.7, Vec2::new(8.0, 6.0), 1.5)
                    .iter()
                    .zip(worley_3d(size, depth, 5, 0.7, 12.0, 0.5))
                    .map(|(&perlin, worley)| map_range(perlin, -(1.0 - worley), 1.0, 0.0, 1.0))
                    .collect::<Vec<f32>>(),
            );

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

    println!("Noise generation took: {:?}", start.elapsed());
    commands.insert_resource(NoiseTextures {
        base: images.add(base_texture),
        detail: images.add(detail_texture),
    });
}
