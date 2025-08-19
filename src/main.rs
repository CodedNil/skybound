#![feature(portable_simd, default_field_values)]

use bevy::{app::plugin_group, prelude::*};

mod render;
use crate::render::WorldRenderingPlugin;

mod debugtext;
use crate::debugtext::DebugTextPlugin;
mod camera;
use crate::camera::CameraPlugin;
pub mod world;
use crate::world::WorldPlugin;

plugin_group! {
    struct CustomPlugins {
        bevy::app:::PanicHandlerPlugin,
        bevy::log:::LogPlugin,
        bevy::app:::TaskPoolPlugin,
        bevy::diagnostic:::FrameCountPlugin,
        bevy::time:::TimePlugin,
        bevy::transform:::TransformPlugin,
        bevy::diagnostic:::DiagnosticsPlugin,
        bevy::input:::InputPlugin,
        bevy::app:::ScheduleRunnerPlugin,
        bevy::window:::WindowPlugin,
        bevy::a11y:::AccessibilityPlugin,
        bevy::app:::TerminalCtrlCHandlerPlugin,
        bevy::asset:::AssetPlugin,
        bevy::winit:::WinitPlugin,
        bevy::render:::RenderPlugin,
        bevy::image:::ImagePlugin,
        bevy::render::pipelined_rendering:::PipelinedRenderingPlugin,
        bevy::core_pipeline:::CorePipelinePlugin,
        bevy::sprite:::SpritePlugin,
        bevy::sprite_render:::SpriteRenderingPlugin,
        bevy::text:::TextPlugin,
        bevy::ui:::UiPlugin,
        bevy::ui_render:::UiRenderPlugin,
    }
}

fn main() {
    App::new()
        .add_plugins((
            CustomPlugins,
            WorldPlugin,
            CameraPlugin,
            WorldRenderingPlugin,
            DebugTextPlugin,
        ))
        .run();
}
