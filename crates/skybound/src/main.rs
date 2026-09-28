mod camera;
mod debugtext;
mod render;
mod ships;
mod show_prepass;
mod world;

use crate::{
    camera::CameraPlugin, debugtext::DebugTextPlugin, render::WorldRenderingPlugin,
    ships::player::PlayerPlugin, show_prepass::ShowPrepassPlugin, world::WorldPlugin,
};
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
                file_path: "../../assets".to_owned(),
                ..default()
            }),
            WorldPlugin,
            PlayerPlugin,
            CameraPlugin,
            WorldRenderingPlugin,
            DebugTextPlugin,
            ShowPrepassPlugin,
        ))
        .run();
}
