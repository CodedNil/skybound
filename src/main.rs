use bevy::{
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use std::env;

mod benchmark;
mod game;
mod graphics;
mod math;

fn main() {
    let mut app = App::new();
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
