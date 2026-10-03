use crate::game::camera::CameraController;
use bevy::{
    camera::RenderTarget,
    diagnostic::DiagnosticsStore,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, save_to_disk},
    },
    window::PresentMode,
};
use std::env;

#[derive(Resource)]
struct BenchmarkTarget(Option<Handle<Image>>);

pub struct BenchmarkPlugin;
impl Plugin for BenchmarkPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderDiagnosticsPlugin)
            .add_systems(PostStartup, setup)
            .add_systems(Update, measure);
    }
}

fn setup(
    mut commands: Commands,
    mut window: Single<&mut Window>,
    mut camera: Single<(Entity, &mut Camera), With<Camera3d>>,
    mut time: ResMut<Time<Virtual>>,
    mut images: ResMut<Assets<Image>>,
) {
    window.present_mode = PresentMode::AutoNoVsync;
    window.resolution.set(1280.0, 720.0);
    if !env::args().any(|arg| arg == "--live") {
        time.pause();
    }
    let (entity, camera) = &mut *camera;
    camera.viewport = None;
    let target = (!env::args().any(|arg| arg == "--window")).then(|| {
        images.add(Image::new_target_texture(
            1280,
            720,
            TextureFormat::Bgra8UnormSrgb,
            None,
        ))
    });
    if let Some(target) = &target {
        commands
            .entity(*entity)
            .insert(RenderTarget::Image(target.clone().into()));
    }
    commands.insert_resource(BenchmarkTarget(target));
    if !env::args().any(|arg| arg == "--live") {
        commands.entity(*entity).insert(CameraController {
            speed: 0.0,
            sensitivity: 0.0,
        });
    }
}

fn measure(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    target: Res<BenchmarkTarget>,
    mut samples: Local<Vec<f64>>,
    mut captured: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    let elapsed = time.elapsed_secs_f64();
    if elapsed < 10.0 {
        return;
    }
    if !*captured {
        let path = env::var("SKYBOUND_SCREENSHOT")
            .unwrap_or_else(|_| "/tmp/skybound-benchmark.png".to_owned());
        commands
            .spawn(
                target
                    .0
                    .clone()
                    .map_or_else(Screenshot::primary_window, Screenshot::image),
            )
            .observe(save_to_disk(path));
        *captured = true;
        return;
    }
    samples.push(time.delta_secs_f64() * 1000.0);
    if elapsed < 30.0 {
        return;
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95 = samples[(samples.len() - 1) * 95 / 100];
    println!(
        "BENCHMARK frames={} mean_ms={mean:.2} p95_ms={p95:.2} fps={:.2}",
        samples.len(),
        1000.0 / mean
    );
    for diagnostic in diagnostics
        .iter()
        .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
    {
        if let Some(ms) = diagnostic.average() {
            println!("GPU {} {ms:.3} ms", diagnostic.path().as_str());
        }
    }
    exit.write(AppExit::Success);
}
