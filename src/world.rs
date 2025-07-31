use crate::camera::CameraController;
use bevy::{
    anti_aliasing::taa::TemporalAntiAliasing,
    camera::Exposure,
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    pbr::{CascadeShadowConfigBuilder, NotShadowCaster, light_consts::lux},
    prelude::*,
};
use std::f32::consts::FRAC_PI_2;

// --- Constants ---
pub const PLANET_RADIUS: f32 = 500_000.0;
const POLE_HEIGHT: f32 = 1_000_000.0;
const POLE_WIDTH: f32 = 10_000.0;
const CAMERA_RESET_THRESHOLD: f32 = 50_000.0;

// Sun Angle Configuration
const MIN_SUN_ELEVATION_DEG: f32 = -15.0;
const MAX_SUN_ELEVATION_DEG: f32 = 70.0;

// --- Components ---
#[derive(Component)]
struct Planet;

#[derive(Component, Default)]
pub struct WorldCoordinates {
    pub latitude: f32,
    pub longitude: f32,
    pub altitude: f32,
}

#[derive(Component)]
struct PoleMarker {
    is_north: bool,
}

#[derive(Component)]
pub struct SunLight;

#[derive(Resource, Default)]
struct PlanetState {
    last_camera_pos: Option<Vec3>,
}

// --- Plugin ---
pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlanetState>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

// --- Systems ---
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Msaa::Off,
        TemporalAntiAliasing::default(),
        Transform::from_xyz(0.0, 4.0, 12.0).looking_at(Vec3::Y * 4.0, Vec3::Y),
        Exposure::SUNLIGHT,
        Bloom::NATURAL,
        DepthPrepass,
        WorldCoordinates::default(),
        CameraController {
            speed: 40.0,
            sensitivity: 0.005,
            yaw: 0.0,
            pitch: 0.0,
        },
    ));

    // Spawn the planet entity
    commands.spawn((
        Transform {
            rotation: Quat::from_rotation_x(-FRAC_PI_2),
            translation: Vec3::new(0.0, -PLANET_RADIUS, 0.0),
            ..default()
        },
        Planet,
    ));

    // Pole markers
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(POLE_WIDTH, POLE_HEIGHT))),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.5, 1.0))),
        NotShadowCaster,
        PoleMarker { is_north: true },
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(POLE_WIDTH, POLE_HEIGHT))),
        MeshMaterial3d(materials.add(Color::srgb(1.0, 0.5, 0.0))),
        NotShadowCaster,
        PoleMarker { is_north: false },
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: lux::DIRECT_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 0.3,
            maximum_distance: 3.0,
            ..default()
        }
        .build(),
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(-Vec3::Y, Vec3::Y),
        SunLight,
    ));

    // Aur Light
    // commands.spawn((
    //     DirectionalLight {
    //         illuminance: lux::DIRECT_SUNLIGHT,
    //         shadows_enabled: true,
    //         ..default()
    //     },
    //     CascadeShadowConfigBuilder {
    //         first_cascade_far_bound: 0.3,
    //         maximum_distance: 3.0,
    //         ..default()
    //     }
    //     .build(),
    //     Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::Y, Vec3::Y),
    // ));
}

fn update(
    mut planet_state: ResMut<PlanetState>,
    mut camera_query: Query<(&mut Transform, &mut WorldCoordinates), With<Camera>>,
    mut planet_query: Query<&mut Transform, (With<Planet>, Without<Camera>)>,
    mut poles_query: Query<
        (&mut Transform, &PoleMarker),
        (With<PoleMarker>, Without<Camera>, Without<Planet>),
    >,
    mut sun_query: Query<
        (&mut Transform, &mut DirectionalLight),
        (
            With<SunLight>,
            Without<Camera>,
            Without<Planet>,
            Without<PoleMarker>,
        ),
    >,
) {
    let (mut camera_transform, mut world_coords) = match camera_query.single_mut() {
        Ok(res) => res,
        Err(_) => return,
    };

    // --- Camera Snapping Logic ---
    let mut snap_delta = Vec3::ZERO;
    if camera_transform.translation.x.abs() > CAMERA_RESET_THRESHOLD
        || camera_transform.translation.z.abs() > CAMERA_RESET_THRESHOLD
    {
        snap_delta = Vec3::new(
            -camera_transform.translation.x,
            0.0,
            -camera_transform.translation.z,
        );
        camera_transform.translation.x = 0.0;
        camera_transform.translation.z = 0.0;
    }

    // Roll the planet under the camera
    let mut planet_transform = planet_query.single_mut().unwrap();
    if let Some(mut previous_pos) = planet_state.last_camera_pos {
        previous_pos += snap_delta;
        let delta_xz = camera_transform.translation.xz() - previous_pos.xz();
        if delta_xz.length_squared() > f32::EPSILON {
            let roll = Quat::from_axis_angle(
                Vec3::new(-delta_xz.y, 0.0, delta_xz.x).normalize(),
                delta_xz.length() / PLANET_RADIUS,
            );
            planet_transform.rotation = roll * planet_transform.rotation;
        }
    }
    planet_transform.translation = Vec3::new(
        camera_transform.translation.x,
        -PLANET_RADIUS,
        camera_transform.translation.z,
    );
    planet_state.last_camera_pos = Some(camera_transform.translation);

    // Compute camera’s latitude/longitude from the planet’s “up” vector
    let v = Vec3::new(0.0, camera_transform.translation.y + PLANET_RADIUS, 0.0);
    let v_local = planet_transform.rotation.conjugate().mul_vec3(v);
    let r = v_local.length();
    world_coords.latitude = (v_local.y / r).clamp(-1.0, 1.0).asin().to_degrees();
    // Handle pole case where x and z are near zero to avoid undefined atan2
    world_coords.longitude = if v_local.x.abs() < f32::EPSILON && v_local.z.abs() < f32::EPSILON {
        0.0
    } else {
        v_local.x.atan2(v_local.z).to_degrees()
    };
    world_coords.altitude = r - PLANET_RADIUS;

    // Snap each pole onto the sphere’s surface and orient it along the normal
    for (mut pole_tf, pole_marker) in &mut poles_query {
        let sign = if pole_marker.is_north { 1.0 } else { -1.0 };
        let world_normal = planet_transform
            .rotation
            .mul_vec3(Vec3::Y * sign)
            .normalize();
        pole_tf.translation =
            planet_transform.translation + world_normal * (PLANET_RADIUS + POLE_HEIGHT * 0.5);
        pole_tf.rotation = Quat::from_rotation_arc(Vec3::Y, world_normal);
    }

    // Rotate the sun light so it's coming from the closest pole
    let (mut sun_transform, mut sun_light) = sun_query.single_mut().unwrap();

    let latitude_abs = (world_coords.latitude.abs() / 90.0).clamp(0.0, 1.0);
    let camera_up = (camera_transform.translation - planet_transform.translation).normalize();
    let pole_sign = (world_coords.latitude >= 0.0) as u8 as f32 * 2.0 - 1.0;
    let planet_pole_direction = planet_transform.rotation.mul_vec3(Vec3::Y) * pole_sign;

    // Project the pole direction onto the camera's horizon to get the sun's azimuth.
    let sun_azimuth = (planet_pole_direction - camera_up * planet_pole_direction.dot(camera_up))
        .normalize_or((Vec3::X - camera_up * Vec3::X.dot(camera_up)).normalize());

    // Linearly interpolate the sun's elevation based on latitude.
    let desired_elevation_rad = (MIN_SUN_ELEVATION_DEG
        + (MAX_SUN_ELEVATION_DEG - MIN_SUN_ELEVATION_DEG) * (world_coords.latitude.abs() / 90.0))
        .to_radians();

    // Point the light and set its intensity.
    sun_transform.rotation = Quat::from_rotation_arc(
        Vec3::NEG_Z,
        -sun_azimuth * desired_elevation_rad.cos() - camera_up * desired_elevation_rad.sin(),
    );
    sun_light.illuminance = lux::DIRECT_SUNLIGHT * latitude_abs;
}
