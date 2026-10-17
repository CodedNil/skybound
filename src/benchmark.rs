use bevy::math::{DQuat, DVec3};
use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::time::Duration;
use std::{env, fs, io};

use bevy::camera::RenderTarget;
use bevy::diagnostic::DiagnosticsStore;
use bevy::ecs::system::SystemParam;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::window::PresentMode;
use bevy::winit::WinitSettings;
use skybound::scene::aur::lightning::events as lightning_events;
use skybound::scene::aur::{EJECTION_CELL, ejection, ejection_fill};
use skybound::scene::polar::{REACTION_RADIUS, light_position};
use skybound::scene::{
    PLANET_RADIUS, ViewUniform,
    planet::{surface_frame, surface_point},
};

use crate::game::camera::CameraController;
use crate::game::flight::{ControlMode, Flight, Mount, Player};
use crate::game::weather::WeatherSimulation;
use crate::game::world::{WorldData, WorldSystems};
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
            .init_resource::<BenchmarkRun>()
            .add_systems(PostStartup, setup)
            .add_systems(Update, (measure, fly_route))
            .add_systems(PostUpdate, replay_route.before(WorldSystems::Rebase));
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
    run: ResMut<'w, BenchmarkRun>,
    weather: Res<'w, WeatherSimulation>,
}

#[expect(
    clippy::too_many_lines,
    reason = "benchmark scene and camera initialization"
)]
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
        mut run,
        weather,
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
        "climate-orbit" => Some(0.5),
        "pole-distant" => Some(0.9),
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
    if !env::args().any(|arg| arg == "--live" || arg == "--animate" || arg == "--route") {
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
    if let Some(cloud_view) = cloud_camera(&scene, &world, &weather) {
        **transform = cloud_view;
    }
    if env::args().any(|arg| arg == "--route") {
        run.route = Some(Route {
            origin: world.position(transform.translation),
            rotation: world.basis * transform.rotation.as_dquat(),
        });
    }
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

fn cloud_camera(scene: &str, world: &WorldData, weather: &WeatherSimulation) -> Option<Transform> {
    if !scene.starts_with("cloud-") {
        return None;
    }
    if scene == "cloud-storm" {
        let storm = weather
            .0
            .cyclones
            .iter()
            .max_by(|a, b| a.intensity.total_cmp(&b.intensity))?;
        let center = storm.center * (PLANET_RADIUS + 12_000.0);
        let frame = surface_frame(center);
        return Some(
            Transform::from_translation(
                world.local(
                    (center + storm.center * 160_000.0 - frame.y_axis * 120_000.0).as_dvec3(),
                ),
            )
            .looking_at(
                world.local(center.as_dvec3()),
                world.rotate_to_local(frame.y_axis),
            ),
        );
    }
    let origin = world.position(Vec3::ZERO).as_vec3();
    let cloud = weather
        .0
        .clouds
        .iter()
        .filter(|cloud| scene == "cloud-high" || cloud.appearance.x > 0.0003)
        .filter(|cloud| {
            if scene == "cloud-high" {
                cloud.up.w < 1200.0
            } else if scene == "cloud-tower" {
                cloud.up.w > 6000.0
            } else {
                cloud.up.w > 2000.0 && cloud.up.w < 6000.0
            }
        })
        .min_by(|a, b| {
            a.center
                .truncate()
                .distance_squared(origin)
                .total_cmp(&b.center.truncate().distance_squared(origin))
        })?;
    let center = cloud.center.truncate();
    let up = cloud.up.truncate();
    let east = cloud.east.truncate();
    let position = match scene {
        "cloud-interior" => center + east * cloud.center.w * 0.15,
        "cloud-top" | "cloud-high" => {
            center + up * (cloud.up.w + cloud.center.w * 1.2) - east * cloud.center.w
        }
        "cloud-edge" => center - up * cloud.up.w * 0.5 - east * cloud.center.w * 1.1,
        "cloud-tower" => center + up * (cloud.up.w + 30_000.0) - east * cloud.center.w * 3.0,
        _ => {
            center
                + (light_position(center) - center).normalize() * cloud.center.w * 2.8
                + up * cloud.up.w * 0.2
        }
    };
    let target = if scene == "cloud-interior" {
        position + east * cloud.center.w
    } else {
        center
    };
    Some(
        Transform::from_translation(world.local(position.as_dvec3()))
            .looking_at(world.local(target.as_dvec3()), world.rotate_to_local(up)),
    )
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
    } else if scene == "planet-orbit" || scene == "climate-orbit" {
        Transform::from_translation(mount + Vec3::Y * 1000.0)
            .looking_at(world.local(DVec3::ZERO), Vec3::Y)
    } else if scene == "pole-distant" {
        Transform::from_translation(mount + Vec3::Y * 1000.0).looking_at(
            world.local(DVec3::Z * f64::from(REACTION_RADIUS)),
            world.up(mount),
        )
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
        "planet-orbit" | "pole-distant" => 350_000.0,
        "climate-orbit" => 476_000.0,
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
        | "polar" | "polar-reaction" | "pole-distant" => Some(Vec4::ZERO),
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
    if lightning {
        let origin = world.position(Vec3::ZERO).as_vec3();
        if let Some(event) = lightning_events(phase)
            .filter(|event| event.strength > 0.2)
            .min_by(|a, b| {
                a.position
                    .distance_squared(origin)
                    .total_cmp(&b.position.distance_squared(origin))
            })
        {
            let basis = surface_frame(event.up);
            let point = event.up * (PLANET_RADIUS + event.reach * 0.4);
            let direction = world.rotate_to_local(basis.y_axis);
            let position = world.local(point.as_dvec3()) - direction * 24_000.0;
            let up = world.up(position);
            mount.translation = position;
            mount.rotation = Transform::default().looking_to(direction, up).rotation;
            player.translation = position + up * 5.8;
            player.rotation = mount.rotation;
            return direction;
        }
    }
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
struct GpuSamples {
    values: Vec<f64>,
    last: Option<Instant>,
}

struct Route {
    origin: DVec3,
    rotation: DQuat,
}

#[derive(Resource)]
struct BenchmarkRun {
    samples: Vec<f64>,
    captured: u32,
    started: Option<f64>,
    warmup: f64,
    duration: f64,
    capture: bool,
    sequence: bool,
    gpu: BTreeMap<String, GpuSamples>,
    route: Option<Route>,
}

impl Default for BenchmarkRun {
    fn default() -> Self {
        let seconds = |name, default| {
            option(name).map_or(default, |s| {
                let value = s.parse::<f64>().expect("benchmark seconds");
                assert!(
                    value.is_finite() && value > 0.0,
                    "benchmark seconds must be positive and finite"
                );
                value
            })
        };
        Self {
            samples: Vec::new(),
            captured: 0,
            started: None,
            warmup: seconds("--warmup", 10.0),
            duration: seconds("--duration", 20.0),
            capture: !env::args().any(|arg| arg == "--no-capture"),
            sequence: env::args().any(|arg| arg == "--animate" || arg == "--capture-sequence"),
            gpu: BTreeMap::new(),
            route: None,
        }
    }
}

fn measure(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    target: Res<BenchmarkTarget>,
    captures: Res<BenchmarkCaptures>,
    mut run: ResMut<BenchmarkRun>,
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
    if elapsed < run.warmup {
        return;
    }
    let sequence = run.sequence;
    if run.capture
        && run.captured < if sequence { 3 } else { 1 }
        && elapsed >= run.warmup + f64::from(run.captured) * 2.0
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
    if elapsed < run.warmup + run.duration {
        run.samples.push(time.delta_secs_f64() * 1000.0);
        for diagnostic in diagnostics
            .iter()
            .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
        {
            if let Some(measurement) = diagnostic.measurement() {
                let samples = run
                    .gpu
                    .entry(diagnostic.path().as_str().to_owned())
                    .or_default();
                if samples.last != Some(measurement.time) {
                    samples.values.push(measurement.value);
                    samples.last = Some(measurement.time);
                }
            }
        }
    }
    if elapsed < run.warmup + run.duration {
        return;
    }
    if captures.0 < run.captured {
        if elapsed < run.warmup + run.duration + 60.0 {
            return;
        }
        error!("BENCHMARK failed: screenshot readback did not complete");
        exit.write(AppExit::error());
        return;
    }
    if run.samples.is_empty() {
        error!("BENCHMARK failed: no frame samples");
        exit.write(AppExit::error());
        return;
    }
    match report(&mut run) {
        Ok(()) => {
            exit.write(AppExit::Success);
        }
        Err(error) => {
            error!("BENCHMARK failed to write results: {error}");
            exit.write(AppExit::error());
        }
    }
}

fn report(run: &mut BenchmarkRun) -> io::Result<()> {
    let samples = &mut run.samples;
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95 = samples[(samples.len() - 1) * 95 / 100];
    println!(
        "BENCHMARK frames={} mean_ms={mean:.2} p95_ms={p95:.2} fps={:.2}",
        samples.len(),
        1000.0 / mean
    );
    let mut report = serde_json::json!({
        "scene": option("--scene").unwrap_or_else(|| "garden".to_owned()),
        "width": resolution().x, "height": resolution().y,
        "frames": samples.len(), "mean_ms": mean, "p95_ms": p95, "fps": 1000.0 / mean,
        "capture": run.capture, "route": run.route.is_some(), "passes": {},
    });
    for (name, samples) in &mut run.gpu {
        samples.values.sort_by(f64::total_cmp);
        let mean = samples.values.iter().sum::<f64>() / samples.values.len() as f64;
        let p95 = samples.values[(samples.values.len() - 1) * 95 / 100];
        println!("GPU {name} {mean:.3} ms p95={p95:.3} ms");
        report["passes"][name] =
            serde_json::json!({"mean_ms": mean, "p95_ms": p95, "samples": samples.values.len()});
    }
    println!("BENCHMARK_JSON {report}");
    if let Some(path) = option("--output") {
        fs::write(
            path,
            serde_json::to_vec_pretty(&report).expect("serializable benchmark results"),
        )?;
    }
    Ok(())
}

fn replay_route(
    time: Res<Time<Real>>,
    run: Res<BenchmarkRun>,
    world: Res<WorldData>,
    mut camera: Single<&mut Transform, With<Camera3d>>,
) {
    let (Some(route), Some(started)) = (&run.route, run.started) else {
        return;
    };
    let seconds = time.elapsed_secs_f64() - started;
    let phase = seconds * TAU / 30.0;
    let offset = DVec3::new(
        phase.sin() * 2500.0,
        (1.0 - (phase * 0.5).cos()) * 2500.0,
        -seconds * 160.0,
    );
    camera.translation = world.local(route.origin + route.rotation * offset);
    let rotation = route.rotation * DQuat::from_rotation_y(phase.sin() * 0.35);
    camera.rotation = (world.basis.conjugate() * rotation).as_quat();
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
