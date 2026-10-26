use bevy::{
    anti_alias::dlss::DlssProjectId,
    asset::uuid::Uuid,
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use std::env;

mod benchmark;
mod game;
mod graphics;
mod math;

/// Identifies the game to NVIDIA's DLSS runtime.
const DLSS_PROJECT: Uuid = Uuid::from_u128(0x106e_92de_300b_4087_a6c3_92da_af7b_284c);

fn main() {
    let mut app = App::new();
    // DLSS sets up the renderer, so it must know the game before the default plugins start it.
    app.insert_resource(DlssProjectId(DLSS_PROJECT));
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: PresentMode::Fifo,
                ..default()
            }),
            ..default()
        }),
        game::GamePlugin,
        graphics::GraphicsPlugin,
    ));
    if env::args().any(|arg| arg == "--benchmark") {
        app.add_plugins(benchmark::BenchmarkPlugin);
    }
    app.run();
}
