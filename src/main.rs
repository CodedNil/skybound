#![feature(portable_simd, default_field_values)]

use bevy::prelude::*;

mod render;
use crate::render::WorldRenderingPlugin;

mod debugtext;
use crate::debugtext::DebugTextPlugin;
mod camera;
use crate::camera::CameraPlugin;
pub mod world;
use crate::world::WorldPlugin;

/// Entry point: builds and runs the Bevy app with plugins.
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WorldPlugin,
            CameraPlugin,
            WorldRenderingPlugin,
            DebugTextPlugin,
        ))
        .run();
}
