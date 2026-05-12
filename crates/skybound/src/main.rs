use bevy::{anti_alias::dlss::DlssProjectId, asset::uuid::uuid, prelude::*};

mod render;
use crate::render::WorldRenderingPlugin;
mod debugtext;
use crate::debugtext::DebugTextPlugin;
mod camera;
use crate::camera::CameraPlugin;
mod ships;
use crate::ships::player::PlayerPlugin;
mod world;
use crate::world::WorldPlugin;
mod show_prepass;
use crate::show_prepass::ShowPrepassPlugin;

fn main() {
    App::new()
        .insert_resource(DlssProjectId(uuid!("32ea1d20-cc8c-4459-9766-595f657a785b")))
        .add_plugins((
            DefaultPlugins.set(AssetPlugin {
                file_path: "../../assets".to_string(),
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
