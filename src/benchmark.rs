use bevy::math::DVec3;
use std::env;
use std::f64::consts::FRAC_PI_2;
use std::time::Duration;

use bevy::camera::RenderTarget;
use bevy::diagnostic::DiagnosticsStore;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::window::PresentMode;
use bevy::winit::WinitSettings;
use skybound::scene::aur::{EJECTION_CELL, ejection, ejection_fill};
use skybound::scene::polar::REACTION_RADIUS;
use skybound::scene::{
    PLANET_RADIUS, ViewUniform,
    planet::{surface_frame, surface_point},
};

use crate::game::camera::CameraController;
use crate::game::flight::{ControlMode, Flight, Mount, Player};
use crate::game::world::WorldData;
use crate::graphics::AtmosphereSettings;

#[derive(Resource)]
struct BenchmarkTarget(Option<Handle<Image>>);

#[derive(Resource, Default)]
struct BenchmarkCaptures(u32);

pub struct BenchmarkPlugin;
impl Plugin for BenchmarkPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderDiagnosticsPlugin)
            .init_resource::<BenchmarkCaptures>()
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
    clock: ResMut<'w, Time>,
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
        mut clock,
        mut images,
        mut world,
    } = params;
    commands.insert_resource(WinitSettings::continuous());
    let resolution = resolution();
    let scene = option("--scene").unwrap_or_else(|| "garden".to_owned());
    let altitude = scene_altitude(&scene);
    mount.translation.z = altitude;
    player.translation.z = altitude + 5.8;
    let latitude = match scene.as_str() {
        "equator" => Some(0.0),
        "sunrise" => Some(f64::from((PLANET_RADIUS / REACTION_RADIUS).asin()) + 0.05),
        "polar" => Some(1.4),
        "polar-reaction" | "polar-beam" => Some(1.1),
        "polar-orbit" | "planet-orbit" => Some(1.53),
        _ => None,
    };
    if let Some(latitude) = latitude {
        *world = WorldData::at(latitude, -FRAC_PI_2);
    }
    if let Some(weather) = scene_weather(&scene) {
        settings.0 = weather;
    }
    let phase = option("--time").map_or(0.0, |s| s.parse::<f32>().expect("simulation time"));
    *time = Time::default();
    time.advance_by(Duration::from_secs_f32(phase));
    time.advance_by(Duration::ZERO);
    *clock = time.as_generic();
    let direction = if scene == "aur-ejection" || scene == "aur-lightning" {
        focus_aur(
            phase,
            &world,
            &mut mount,
            &mut player,
            scene == "aur-lightning",
        )
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
    if scene == "nae-close" {
        flight.mode = ControlMode::Flying;
    }
    **transform = scene_camera(
        &scene,
        &world,
        mount.translation,
        player.translation,
        direction,
        phase,
    );
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

fn scene_camera(
    scene: &str,
    world: &WorldData,
    mount: Vec3,
    player: Vec3,
    direction: Vec3,
    phase: f32,
) -> Transform {
    if scene == "cloud-storm" {
        storm_camera(world, mount, phase)
    } else if scene == "cloud-bank" {
        cloud_bank_camera(world, mount, phase)
    } else if scene == "cloud-top" {
        Transform::from_translation(mount - direction * 5000.0 + Vec3::Z * 2500.0)
            .looking_at(mount + direction * 12_000.0 - Vec3::Z * 20_000.0, Vec3::Z)
    } else if scene == "nae-close" {
        Transform::from_translation(player + vec3(1.5, -4.0, 0.8)).looking_at(player, Vec3::Z)
    } else if scene == "aur-overhead" {
        Transform::from_translation(mount + Vec3::Z * 200.0)
            .looking_at(mount.with_z(0.0) + direction * 2000.0, Vec3::Z)
    } else if scene == "planet-orbit" {
        Transform::from_translation(mount).looking_at(world.local(DVec3::ZERO), Vec3::Y)
    } else if scene == "polar-orbit" || scene == "polar-beam" {
        let radius = if scene == "polar-beam" {
            REACTION_RADIUS
        } else {
            PLANET_RADIUS
        };
        Transform::from_translation(mount + vec3(0.0, -1000.0, 100.0))
            .looking_at(world.local(DVec3::Z * f64::from(radius)), Vec3::Z)
    } else if scene == "cloud-tower" {
        Transform::from_translation(mount - direction * 30_000.0 + Vec3::Z * 8000.0)
            .looking_at(mount - Vec3::Z * 12_000.0, Vec3::Z)
    } else {
        Transform::from_translation(mount - direction * 85.0 + Vec3::Z * 18.0)
            .looking_at(mount + Vec3::Z * 5.0, Vec3::Z)
    }
}

fn storm_camera(world: &WorldData, position: Vec3, seconds: f32) -> Transform {
    use skybound::scene::climate::clouds::{STORM, region_size, storm_at};
    let cell_size = region_size(STORM);
    let origin = world.position(position).as_vec3();
    let mut closest = None;
    let mut distance = f32::MAX;
    for x in -7..=7 {
        for y in -7..=7 {
            for z in -7..=7 {
                let point = vec3(x as f32, y as f32, z as f32) * cell_size;
                if let Some(storm) = storm_at(point, seconds) {
                    let candidate = storm.center.distance_squared(origin);
                    if candidate < distance {
                        closest = Some(storm);
                        distance = candidate;
                    }
                }
            }
        }
    }
    let storm = closest.expect("storm benchmark requires a storm");
    let side = (Vec3::Z - storm.up * storm.up.z).normalize_or(Vec3::X);
    let target = world.local(storm.center.as_dvec3());
    let up = world.rotate_to_local(storm.up);
    let camera = world.local((storm.center + side * storm.radius * 4.0).as_dvec3());
    Transform::from_translation(camera).looking_at(target, up)
}

fn cloud_bank_camera(world: &WorldData, position: Vec3, seconds: f32) -> Transform {
    use skybound::scene::climate::{Climate, clouds, weather_at};
    use skybound::scene::noise::{cellular, fbm, rotated};
    let origin = world.position(position).as_vec3();
    let frame = surface_frame(origin);
    let mut target = origin;
    let mut best = -1.0;
    for x in -10..=10 {
        for y in -10..=10 {
            let point =
                (origin + frame.x_axis * (x as f32 * 2000.0) + frame.y_axis * (y as f32 * 2000.0))
                    .normalize()
                    * (PLANET_RADIUS + 11_000.0);
            let climate = Climate::on_planet(point, seconds, vec4(0.95, 0.65, 0.8, 0.9));
            let weather = weather_at(clouds::region_coordinates(clouds::column_point(point), 0));
            let coverage =
                clouds::region_value(weather.x, seconds).clamp(0.0, 1.0) * climate.low_cloud;
            let sample = |p| {
                let q = rotated(p);
                let shape = |p| 1.0 - (1.0 - fbm(p, 4, 0x11)) * cellular(p, 4, 0xd3);
                vec4(
                    0.8,
                    shape(p).midpoint(shape(q)),
                    fbm(p, 4, 0x35).midpoint(fbm(q, 4, 0x35)),
                    fbm(p, 4, 0x61).midpoint(fbm(q, 4, 0x61)),
                )
            };
            let flow = clouds::flow(clouds::column_point(point), seconds, 0);
            let mut noise = sample(flow.previous).lerp(sample(flow.current), flow.blend);
            noise.z = weather.w * 0.9375 + noise.z * 0.0625;
            let density = clouds::layer_density(11_000.0, coverage, 0, noise, 0.0);
            if density > best {
                best = density;
                target = point;
            }
        }
    }
    let camera = target - frame.y_axis * 25_000.0 - frame.z_axis * 7000.0;
    Transform::from_translation(world.local(camera.as_dvec3())).looking_at(
        world.local(target.as_dvec3()),
        world.rotate_to_local(frame.z_axis),
    )
}

fn scene_altitude(scene: &str) -> f32 {
    match scene {
        "cloud-bank" => 11_000.0,
        "cloud-top" => 25_000.0,
        "cloud-edge" => 6350.0,
        "cloud-high" => 48_000.0,
        "cloud-tower" | "cloud-storm" => 78_000.0,
        "cloud-interior" => 7500.0,
        "veil" => 25.0,
        "aur-horizon" => 12_000.0,
        "aur-overhead" => 4000.0,
        "polar-orbit" => 77_559.0,
        "planet-orbit" => 350_000.0,
        "polar-beam" => 450_000.0,
        "aur-ejection" | "aur-lightning" => 650.0,
        "garden" | "clear" | "mounted" | "nae-close" | "equator" | "sunrise" | "polar"
        | "polar-reaction" => 2400.0,
        _ => panic!("unknown benchmark scene"),
    }
}

fn scene_weather(scene: &str) -> Option<Vec4> {
    match scene {
        "cloud-bank" => Some(vec4(0.95, 0.65, 0.8, 0.9)),
        "cloud-storm" => Some(Vec4::splat(0.95)),
        "cloud-high" => Some(vec4(0.1, 0.1, 0.95, 0.3)),
        "cloud-tower" => Some(vec4(0.95, 0.95, 0.7, 0.95)),
        "cloud-edge" | "cloud-interior" => Some(vec4(0.95, 0.85, 0.3, 0.8)),
        "aur-horizon" | "aur-overhead" | "aur-ejection" | "clear" | "equator" | "sunrise"
        | "polar" | "polar-reaction" => Some(Vec4::ZERO),
        _ => None,
    }
}

fn focus_aur(
    phase: f32,
    world: &WorldData,
    mount: &mut Transform,
    player: &mut Transform,
    lightning: bool,
) -> Vec3 {
    let mut view = ViewUniform {
        planet_center_value: world.local(DVec3::ZERO).extend(0.0),
        planet_rotation: world.basis.conjugate().as_quat(),
        ..default()
    };
    view.times.x = phase
        + if env::args().any(|arg| arg == "--animate") {
            10.0
        } else {
            0.0
        };
    view.prepare_flashes();
    let cell = (world.surface_position(Vec3::ZERO).truncate() / EJECTION_CELL).floor();
    let event = (0..49)
        .map(|index| {
            ejection(
                cell + vec2((index / 7) as f32 - 3.0, (index % 7) as f32 - 3.0),
                view.time(),
            )
        })
        .max_by(|a, b| {
            let score = |event: &[Vec4; 2]| event[0].w * ejection_fill(0.5, event[0].z);
            score(a).total_cmp(&score(b))
        })
        .expect("ejection cells");
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
    let altitude = if lightning {
        world.altitude(mount.translation) as f32
    } else {
        event[1].w * 0.25 + 1000.0
    };
    let point = surface_point(center.extend(altitude));
    let direction = if lightning {
        Vec3::Y
    } else {
        world.rotate_to_local(surface_frame(point) * vec3(-event[1].y, -event[1].x, 0.0))
    };
    let distance = if lightning { 4000.0 } else { 36_000.0 };
    let position = world.local(point.as_dvec3()) - direction * distance;
    let up = world.up(position);
    mount.translation = position;
    mount.rotation = Transform::default().looking_to(direction, up).rotation;
    player.translation = position + up * 5.8;
    player.rotation = mount.rotation;
    direction
}

#[derive(Default)]
struct BenchmarkRun {
    samples: Vec<f64>,
    captured: u32,
    started: Option<f64>,
}

fn measure(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    target: Res<BenchmarkTarget>,
    captures: Res<BenchmarkCaptures>,
    mut run: Local<BenchmarkRun>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs_f64();
    if !diagnostics
        .iter()
        .any(|d| d.path().as_str() == "render/volumetrics/elapsed_gpu" && d.average().is_some())
    {
        if now > 60.0 {
            error!("BENCHMARK failed: volumetric pipeline never became ready");
            exit.write(AppExit::error());
        }
        return;
    }
    let elapsed = now - *run.started.get_or_insert(now);
    if elapsed < 10.0 {
        return;
    }
    let sequence = env::args().any(|arg| arg == "--animate");
    if run.captured < if sequence { 3 } else { 1 }
        && elapsed >= 10.0 + f64::from(run.captured) * 2.0
    {
        let mut path = env::var("SKYBOUND_SCREENSHOT")
            .unwrap_or_else(|_| "/tmp/skybound-benchmark.png".to_owned());
        if sequence {
            path = path.replace(".png", &format!("-{}.png", run.captured));
        }
        commands
            .spawn(
                target
                    .0
                    .clone()
                    .map_or_else(Screenshot::primary_window, Screenshot::image),
            )
            .observe(save_to_disk(path))
            .observe(
                |_: On<ScreenshotCaptured>, mut captures: ResMut<BenchmarkCaptures>| {
                    captures.0 += 1;
                },
            );
        run.captured += 1;
        return;
    }
    if elapsed < 30.0 {
        run.samples.push(time.delta_secs_f64() * 1000.0);
    }
    if elapsed < 30.0 {
        return;
    }
    if captures.0 < run.captured {
        if elapsed < 90.0 {
            return;
        }
        error!("BENCHMARK failed: screenshot readback did not complete");
        exit.write(AppExit::error());
        return;
    }
    let samples = &mut run.samples;
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
