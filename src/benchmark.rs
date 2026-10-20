use crate::{
    game::{
        camera::CameraController,
        flight::{ControlMode, Flight, Mount, Player},
        weather::{CloudField, lightning_events},
        world::{WorldData, WorldSystems},
    },
    graphics::AtmosphereSettings,
    math::{ToEngine, ToScene},
};
use bevy::{
    camera::RenderTarget,
    diagnostic::DiagnosticsStore,
    ecs::system::SystemParam,
    math::{DQuat, DVec3},
    platform::time::Instant,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::PresentMode,
    winit::WinitSettings,
};
use skybound::scene::{
    FEATURE_SCALE, PLANET_RADIUS,
    aur::{
        EJECTION_CELL, SPIRE_CELL, ejection, ejection_fill, spire, spire_coordinates, spire_count,
        spire_point,
    },
    climate::{CIRRUS, CUMULUS, CYCLONE, DECK, TOWER},
    planet::{REACTION_RADIUS, surface_frame, surface_point},
};
use spirv_std::glam::Vec4 as SceneVec4;
use std::{
    collections::BTreeMap,
    env,
    f64::consts::{FRAC_PI_2, TAU},
    fs, io,
    time::Duration,
};

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
    );
    if let Some(cloud_view) = cloud_camera(&scene, &world, phase) {
        **transform = cloud_view;
    }
    if scene == "aur-spires" {
        **transform = spire_camera(&world, mount.translation);
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

/// Frame the main spire of the nearest archipelago from low over the veil.
fn spire_camera(world: &WorldData, mount: Vec3) -> Transform {
    let here =
        spire_coordinates(world.position(mount).as_vec3().to_scene() * FEATURE_SCALE).to_engine();
    let origin = (here.truncate() / SPIRE_CELL).floor();
    let cell = (0..40i32)
        .flat_map(|ring| (-ring..=ring).flat_map(move |x| [(x, -ring), (x, ring)]))
        .map(|(x, y)| origin + vec2(x as f32, y as f32))
        .find(|cell| spire_count(cell.to_scene()) > 0)
        .expect("an archipelago near the start");
    let centre = (cell + 0.5) * SPIRE_CELL + spire(cell.to_scene(), 0).base.to_engine();
    let point = |offset: Vec2, altitude: f32| {
        (spire_point((centre + offset).extend(altitude).to_scene()) / FEATURE_SCALE)
            .to_engine()
            .as_dvec3()
    };
    let eye = point(vec2(-13_000.0, -5000.0), 1600.0);
    let target = point(Vec2::ZERO, 2600.0);
    Transform::from_translation(world.local(eye)).looking_at(
        world.local(target),
        world.rotate_to_local(eye.normalize().as_vec3()),
    )
}

/// Frame the nearest strong weather system of the kind a scene shows.
fn cloud_camera(scene: &str, world: &WorldData, time: f32) -> Option<Transform> {
    let kind = match scene {
        "cloud-storm" => CYCLONE,
        "cloud-tower" | "cloud-rain" | "cloud-above" => TOWER,
        "cloud-bank" => DECK,
        "cloud-high" => CIRRUS,
        "cloud-edge" | "cloud-interior" | "cloud-top" => CUMULUS,
        _ => return None,
    };
    let field = CloudField::at(time);
    let start = world.position(Vec3::ZERO).as_vec3().to_scene().normalize();
    let parcel = field
        .parcels
        .iter()
        // Systems nearer the poles, where their light shows their shape.
        .filter(|parcel| {
            // Rain is staged where the pole light is low, so the storm's
            // rain is lit and its bow stands above the horizon.
            let latitude = parcel.centre.z.abs();
            let placed = if scene == "cloud-rain" {
                // Just inside the terminator, where the pole light skims in.
                let dawn = (PLANET_RADIUS / REACTION_RADIUS).asin();
                (dawn + 0.03..dawn + 0.25).contains(&latitude.asin())
                    && parcel.shape.w - parcel.shape.z > 12_000.0
            } else {
                latitude > 0.75
            };
            parcel.kind() == kind && parcel.shape.y > 0.5 && placed
        })
        .max_by(|a, b| {
            a.centre
                .truncate()
                .dot(start)
                .total_cmp(&b.centre.truncate().dot(start))
        })?;
    let up = parcel.centre.truncate();
    let frame = surface_frame(up);
    let (reach, base, top) = (parcel.centre.w, parcel.shape.z, parcel.shape.w);
    // Cloud heights and sizes are in feature space; system reaches in the world.
    let core = parcel.detail.x / FEATURE_SCALE;
    let at = |altitude: f32, north: f32| {
        world.local(
            (up * (PLANET_RADIUS + altitude / FEATURE_SCALE) + frame.y_axis * north)
                .as_dvec3()
                .to_engine(),
        )
    };
    let middle = base + (top - base) * 0.5;
    let (position, target) = match scene {
        "cloud-storm" => (
            at(260_000.0 * FEATURE_SCALE, -reach * 0.9),
            at(10_000.0, 0.0),
        ),
        "cloud-tower" => (at(middle, -core * 5.0), at(middle, 0.0)),
        // From the sunward side, looking into the rain beneath the storm,
        // where its bow stands opposite the light.
        "cloud-rain" => {
            let sunward = core * 3.0 * up.z.signum();
            (
                at(base - 1500.0, sunward * 2.0),
                at(base - 2500.0, -sunward),
            )
        }
        "cloud-bank" => (
            at(base - parcel.detail.z - 1500.0, -reach * 0.3),
            at(base, reach * 0.3),
        ),
        "cloud-high" => (at(base - 6000.0, -reach * 0.2), at(base, reach * 0.4)),
        "cloud-interior" => (at(middle, 0.0), at(middle, 20_000.0 / FEATURE_SCALE)),
        "cloud-top" => (at(top + 6000.0, -reach * 0.8), at(top * 0.8, 0.0)),
        // High above a storm field, looking almost straight down.
        "cloud-above" => (at(top + 30_000.0, -reach * 0.3), at(base, reach * 0.1)),
        _ => (
            at(base + 500.0, -reach * 1.05),
            at(base + (top - base) * 0.4, 0.0),
        ),
    };
    Some(
        Transform::from_translation(position)
            .looking_at(target, world.rotate_to_local(up.to_engine())),
    )
}

fn scene_camera(
    scene: &str,
    world: &WorldData,
    mount: Vec3,
    player: Vec3,
    direction: Vec3,
) -> Transform {
    if scene == "nae-close" {
        Transform::from_translation(player + vec3(1.5, -4.0, 0.8)).looking_at(player, Vec3::Z)
    } else if scene == "aur-high" {
        Transform::from_translation(mount - direction * 3000.0 + Vec3::Z * 1000.0)
            .looking_at(mount + direction * 100_000.0 - Vec3::Z * 60_000.0, Vec3::Z)
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
    } else {
        Transform::from_translation(mount - direction * 85.0 + Vec3::Z * 18.0)
            .looking_at(mount + Vec3::Z * 5.0, Vec3::Z)
    }
}

/// The camera's altitude: in the world for the polar scenes, which frame
/// the pole light, and scaled from feature space for all the rest.
fn scene_altitude(scene: &str) -> f32 {
    let feature = match scene {
        "cloud-bank" => 11_000.0,
        "cloud-top" => 25_000.0,
        "cloud-edge" => 6350.0,
        "cloud-high" => 48_000.0,
        "cloud-tower" | "cloud-storm" | "cloud-above" => 78_000.0,
        "cloud-interior" | "cloud-rain" => 7500.0,
        "veil" => 25.0,
        "aur-horizon" => 12_000.0,
        "aur-high" => 80_000.0,
        "aur-overhead" => 4000.0,
        "aur-skim" => 3820.0,
        "polar-orbit" => return 77_559.0,
        "planet-orbit" | "pole-distant" => return 350_000.0,
        "climate-orbit" => return 476_000.0,
        "polar-beam" => return 450_000.0,
        "aur-ejection" | "aur-lightning" => 650.0,
        "aur-spires" | "garden" | "clear" | "mounted" | "nae-close" | "equator" | "sunrise"
        | "polar" | "polar-reaction" => 2400.0,
        _ => panic!("unknown benchmark scene"),
    };
    feature / FEATURE_SCALE
}

fn scene_weather(scene: &str) -> Option<Vec4> {
    match scene {
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
    let (position, direction) = if lightning {
        let origin = world.position(Vec3::ZERO).as_vec3().to_scene();
        let event = lightning_events(phase)
            .filter(|event| event.strength > 0.2)
            .min_by(|a, b| {
                a.position
                    .distance_squared(origin)
                    .total_cmp(&b.position.distance_squared(origin))
            })
            .expect("planetary lightning activity");
        let direction = world.rotate_to_local(surface_frame(event.up).y_axis.to_engine());
        let point = event.up * (PLANET_RADIUS + event.reach * 0.4);
        (
            world.local(point.as_dvec3().to_engine()) - direction * (24_000.0 / FEATURE_SCALE),
            direction,
        )
    } else {
        let seconds = phase
            + if env::args().any(|arg| arg == "--animate") {
                10.0
            } else {
                0.0
            };
        let cell = (world.surface_position(Vec3::ZERO).truncate().to_scene() * FEATURE_SCALE
            / EJECTION_CELL)
            .floor();
        let event = (0..49)
            .map(|index| {
                ejection(
                    cell + vec2((index / 7) as f32 - 3.0, (index % 7) as f32 - 3.0).to_scene(),
                    seconds,
                )
            })
            .max_by(|a, b| {
                let score = |event: &[SceneVec4; 2]| event[0].w * ejection_fill(0.5, event[0].z);
                score(a).total_cmp(&score(b))
            })
            .expect("ejection cells");
        let center = event[0].truncate().truncate();
        let point = surface_point(center.extend(event[1].w * 0.25 + 1000.0) / FEATURE_SCALE);
        let direction = world.rotate_to_local(
            (surface_frame(point) * vec3(-event[1].y, -event[1].x, 0.0).to_scene()).to_engine(),
        );
        (
            world.local(point.as_dvec3().to_engine()) - direction * (36_000.0 / FEATURE_SCALE),
            direction,
        )
    };
    let up = world.up(position);
    mount.translation = position;
    mount.rotation = Transform::default().looking_to(direction, up).rotation;
    player.translation = position + up * 5.8;
    player.rotation = mount.rotation;
    direction
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
