mod benchmark;
mod game;
mod graphics;

use std::env;

use bevy::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, game::GamePlugin, graphics::GraphicsPlugin));
    if env::args().any(|arg| arg == "--benchmark") {
        app.add_plugins(benchmark::BenchmarkPlugin);
    }
    app.run();
}
