use bevy::{
    anti_alias::dlss::DlssProjectId,
    asset::uuid::Uuid,
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use skybound::scene::life::organs;
use std::env;

mod benchmark;
mod game;
mod graphics;
mod math;
mod snapshot;

/// Identifies the game to NVIDIA's DLSS runtime.
const DLSS_PROJECT: Uuid = Uuid::from_u128(0x106e_92de_300b_4087_a6c3_92da_af7b_284c);

fn main() {
    // Every species' anatomy is laid out in its body once, all at the same time, before anything
    // asks for one.
    organs::prepare();
    // A benchmark draws offscreen and never needs to be seen: its window stays hidden, so
    // profiling leaves the desktop alone, unless `--window` asks to watch it.
    let benchmark = env::args().any(|arg| arg == "--benchmark");
    let hidden = benchmark && !env::args().any(|arg| arg == "--window");
    let mut app = App::new();
    // DLSS sets up the renderer, so it must know the game before the default plugins start it.
    app.insert_resource(DlssProjectId(DLSS_PROJECT));
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: PresentMode::Fifo,
                visible: !hidden,
                focused: !hidden,
                ..default()
            }),
            ..default()
        }),
        game::GamePlugin,
        graphics::GraphicsPlugin,
        snapshot::SnapshotPlugin,
    ));
    if benchmark {
        app.add_plugins(benchmark::BenchmarkPlugin);
    }
    app.run();
}
