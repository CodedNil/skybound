use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin, FrameTimeGraphConfig},
    prelude::*,
};
use std::time::Duration;

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
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont {
                        font: Handle::<Font>::default(),
                        font_size: 32.0,
                        ..Default::default()
                    },
                    text_color: Color::WHITE,
                    enabled: true,
                    refresh_interval: Duration::from_millis(100),
                    frame_time_graph_config: FrameTimeGraphConfig::target_fps(60.0),
                },
            },
        ))
        .run();
}
