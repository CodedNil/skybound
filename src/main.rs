use bevy::{
    anti_alias::dlss::DlssProjectId,
    app::ScheduleRunnerPlugin,
    asset::uuid::Uuid,
    prelude::*,
    window::{ExitCondition, PresentMode, WindowPlugin},
    winit::WinitPlugin,
};
use skybound::scene::life::organs;
use std::{env, time::Duration};

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
    // A benchmark draws offscreen and never needs to be seen: it makes no window at all, so
    // profiling leaves the desktop alone, unless `--window` asks to watch it.
    let benchmark = env::args().any(|arg| arg == "--benchmark");
    let windowless = benchmark && !env::args().any(|arg| arg == "--window");
    let mut app = App::new();
    // DLSS sets up the renderer, so it must know the game before the default plugins start it.
    app.insert_resource(DlssProjectId(DLSS_PROJECT));
    let defaults = DefaultPlugins.set(WindowPlugin {
        primary_window: (!windowless).then(|| Window {
            present_mode: PresentMode::Fifo,
            ..default()
        }),
        // With no window the app must not take that for every window having been closed.
        exit_condition: if windowless {
            ExitCondition::DontExit
        } else {
            ExitCondition::OnAllClosed
        },
        ..default()
    });
    if windowless {
        // Nothing needs the desktop's event loop: frames are driven in a plain loop instead.
        app.add_plugins((
            defaults.disable::<WinitPlugin>(),
            ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        ));
    } else {
        app.add_plugins(defaults);
    }
    app.add_plugins((
        game::GamePlugin,
        graphics::GraphicsPlugin,
        snapshot::SnapshotPlugin,
    ));
    if benchmark {
        app.add_plugins(benchmark::BenchmarkPlugin);
    }
    app.run();
}
