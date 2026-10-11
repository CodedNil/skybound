use std::env;
use std::f32::consts::FRAC_PI_2;
use std::time::Duration;

use bevy::camera::RenderTarget;
use bevy::diagnostic::DiagnosticsStore;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::PresentMode;
use bevy::winit::WinitSettings;
use skybound::scene::aur::{EJECTION_CELL, ejection};
use skybound::scene::{PLANET_RADIUS, ViewUniform};

use crate::game::camera::CameraController;
use crate::game::flight::{ControlMode, Flight, Mount, Player};
use crate::game::world::WorldData;
use crate::graphics::AtmosphereSettings;

#[derive(Resource)]
struct BenchmarkTarget(Option<Handle<Image>>);

pub struct BenchmarkPlugin;
impl Plugin for BenchmarkPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderDiagnosticsPlugin)
            .add_systems(PostStartup, setup)
            .add_systems(Update, (measure, fly_route));
    }
}

type BenchmarkCamera = (With<Camera3d>, Without<Mount>, Without<Player>);
type BenchmarkMount = (With<Mount>, Without<Player>, Without<Camera3d>);
type BenchmarkPlayer = (With<Player>, Without<Mount>, Without<Camera3d>);

#[derive(SystemParam)]
struct BenchmarkSetup<'w, 's> {
    commands: Commands<'w, 's>,
    window: Single<'w, 's, &'static mut Window>,
    camera: Single<'w, 's, (Entity, &'static mut Camera, &'static mut Transform), BenchmarkCamera>,
    mount: Single<'w, 's, &'static mut Transform, BenchmarkMount>,
    player: Single<'w, 's, &'static mut Transform, BenchmarkPlayer>,
    settings: ResMut<'w, AtmosphereSettings>,
    flight: ResMut<'w, Flight>,
    time: ResMut<'w, Time<Virtual>>,
    images: ResMut<'w, Assets<Image>>,
    world: ResMut<'w, WorldData>,
}

fn setup(params: BenchmarkSetup) {
    let BenchmarkSetup {
        mut commands,
        mut window,
        mut camera,
        mut mount,
        mut player,
        mut settings,
        mut flight,
        mut time,
        mut images,
        mut world,
    } = params;
    commands.insert_resource(WinitSettings::continuous());
    let resolution = resolution();
    let scene = option("--scene").unwrap_or_else(|| "garden".to_owned());
    let altitude = match scene.as_str() {
        "cloud-edge" => 6350.0,
        "cloud-high" => 8000.0,
        "cloud-tower" => 26_000.0,
        "cloud-interior" => 7500.0,
        "veil" => 25.0,
        "aur-horizon" => 12_000.0,
        "aur-ejection" | "aur-lightning" => 650.0,
        "garden" | "clear" | "mounted" | "nae-close" | "equator" | "sunrise" | "polar" => 2400.0,
        _ => panic!("unknown benchmark scene"),
    };
    mount.translation.z = altitude;
    player.translation.z = altitude + 5.8;
    let latitude = match scene.as_str() {
        "equator" => Some(0.0),
        "sunrise" => Some(0.87),
        "polar" => Some(1.4),
        _ => None,
    };
    if let Some(latitude) = latitude {
        world.camera_offset = vec3(0.0, PLANET_RADIUS * (FRAC_PI_2 - latitude), 0.0);
    }
    if let Some(weather) = scene_weather(&scene) {
        settings.0 = weather;
    }
    let phase = option("--time").map_or(0.0, |s| s.parse::<f32>().expect("simulation time"));
    time.advance_by(Duration::from_secs_f32(phase));
    let direction = if scene == "aur-ejection" || scene == "aur-lightning" {
        focus_aur(
            phase,
            world.camera_offset,
            &mut mount,
            &mut player,
            scene == "aur-lightning",
        )
    } else if latitude.is_some() {
        -Vec3::Y
    } else {
        Vec3::Y
    };
    println!(
        "BENCHMARK scene={scene} resolution={}x{} simulation_time={phase}",
        resolution.x, resolution.y
    );
    window.present_mode = PresentMode::AutoNoVsync;
    window
        .resolution
        .set(resolution.x as f32, resolution.y as f32);
    if !env::args().any(|arg| arg == "--live" || arg == "--animate") {
        time.pause();
    }
    let (entity, camera, transform) = &mut *camera;
    **transform = if scene == "nae-close" {
        flight.mode = ControlMode::Flying;
        Transform::from_translation(player.translation + vec3(1.5, -4.0, 0.8))
            .looking_at(player.translation, Vec3::Z)
    } else if scene == "cloud-tower" {
        Transform::from_translation(mount.translation - direction * 12_000.0 + Vec3::Z * 8000.0)
            .looking_at(mount.translation - Vec3::Z * 5000.0, Vec3::Z)
    } else {
        Transform::from_translation(mount.translation - direction * 85.0 + Vec3::Z * 18.0)
            .looking_at(mount.translation + Vec3::Z * 5.0, Vec3::Z)
    };
    camera.viewport = None;
    let target = (!env::args().any(|arg| arg == "--window")).then(|| {
        images.add(Image::new_target_texture(
            resolution.x,
            resolution.y,
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

fn scene_weather(scene: &str) -> Option<Vec4> {
    match scene {
        "cloud-high" => Some(vec4(0.1, 0.1, 0.95, 0.3)),
        "cloud-tower" => Some(vec4(0.95, 0.95, 0.7, 0.95)),
        "cloud-edge" | "cloud-interior" => Some(vec4(0.95, 0.85, 0.3, 0.8)),
        "aur-horizon" | "clear" | "equator" | "sunrise" | "polar" => Some(Vec4::ZERO),
        _ => None,
    }
}

fn focus_aur(
    phase: f32,
    offset: Vec3,
    mount: &mut Transform,
    player: &mut Transform,
    lightning: bool,
) -> Vec3 {
    let mut view = ViewUniform {
        camera_position: vec4(0.0, 0.0, offset.x, offset.y),
        ..default()
    };
    view.times.x = phase
        + if env::args().any(|arg| arg == "--animate") {
            10.0
        } else {
            0.0
        };
    view.prepare_flashes();
    let cell = (offset.truncate() / EJECTION_CELL).floor();
    let event = (0..9)
        .map(|index| {
            ejection(
                cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0),
                view.time(),
            )
        })
        .max_by(|a, b| a[0].w.total_cmp(&b[0].w))
        .expect("ejection cells");
    let direction = if lightning {
        Vec3::Y
    } else {
        vec3(-event[1].y, event[1].x, 0.0)
    };
    let center = if lightning {
        view.aur_bolts
            .iter()
            .find(|bolt| bolt[0].w > 0.0)
            .map_or_else(
                || event[0].truncate().truncate(),
                |bolt| bolt[0].truncate().truncate(),
            )
    } else {
        event[0].truncate().truncate()
    };
    let position = (center - offset.truncate()).extend(mount.translation.z) - direction * 1600.0;
    mount.translation = position;
    mount.rotation = Transform::default().looking_to(direction, Vec3::Z).rotation;
    player.translation = position + Vec3::Z * 5.8;
    player.rotation = mount.rotation;
    direction
}

fn measure(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    target: Res<BenchmarkTarget>,
    mut samples: Local<Vec<f64>>,
    mut captured: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    let elapsed = time.elapsed_secs_f64();
    if elapsed < 10.0 {
        return;
    }
    let sequence = env::args().any(|arg| arg == "--animate");
    if *captured < if sequence { 3 } else { 1 } && elapsed >= 10.0 + f64::from(*captured) * 2.0 {
        let mut path = env::var("SKYBOUND_SCREENSHOT")
            .unwrap_or_else(|_| "/tmp/skybound-benchmark.png".to_owned());
        if sequence {
            path = path.replace(".png", &format!("-{}.png", *captured));
        }
        commands
            .spawn(
                target
                    .0
                    .clone()
                    .map_or_else(Screenshot::primary_window, Screenshot::image),
            )
            .observe(save_to_disk(path));
        *captured += 1;
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

fn resolution() -> UVec2 {
    option("--size").map_or(UVec2::new(1280, 720), |s| {
        let (w, h) = s
            .split_once('x')
            .expect("benchmark size must be WIDTHxHEIGHT");
        UVec2::new(w.parse().expect("width"), h.parse().expect("height"))
    })
}

fn option(name: &str) -> Option<String> {
    let mut args = env::args();
    args.find(|arg| arg == name).and_then(|_| args.next())
}

fn fly_route(time: Res<Time>, mut keys: ResMut<ButtonInput<KeyCode>>) {
    if !env::args().any(|arg| arg == "--live") {
        return;
    }
    keys.press(KeyCode::KeyW);
    keys.press(KeyCode::ShiftLeft);
    if time.elapsed_secs() % 8.0 < 4.0 {
        keys.press(KeyCode::Space);
        keys.release(KeyCode::ControlLeft);
    } else {
        keys.release(KeyCode::Space);
        keys.press(KeyCode::ControlLeft);
    }
}
