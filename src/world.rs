use crate::camera::CameraController;
use bevy::{
    anti_aliasing::taa::TemporalAntiAliasing,
    camera::Exposure,
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    pbr::{
        Atmosphere, AtmosphereSettings, CascadeShadowConfigBuilder, NotShadowCaster,
        light_consts::lux,
    },
    prelude::*,
};
use std::f32::consts::FRAC_PI_2;

// --- Constants ---
const PLANET_RADIUS: f32 = 50_000.0;
const POLE_HEIGHT: f32 = 1_000_000.0;
const POLE_WIDTH: f32 = 2000.0;
const ATMOSPHERE_HEIGHT: f32 = 100_000.0;

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
struct SunLight;

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
        Atmosphere {
            bottom_radius: PLANET_RADIUS,
            top_radius: PLANET_RADIUS + ATMOSPHERE_HEIGHT,
            ..Atmosphere::EARTH
        },
        AtmosphereSettings {
            aerial_view_lut_max_distance: 3.2e5,
            scene_units_to_m: 1.0,
            ..Default::default()
        },
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
        Mesh3d(meshes.add(Sphere::new(PLANET_RADIUS).mesh().ico(76).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.0, 0.0, 0.0),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform {
            rotation: Quat::from_rotation_x(-FRAC_PI_2),
            translation: Vec3::new(0.0, -PLANET_RADIUS, 0.0),
            ..default()
        },
        NotShadowCaster,
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
        Mesh3d(meshes.add(Cylinder::new(2.0, 20.0))),
        MeshMaterial3d(materials.add(Color::srgb(1.0, 1.0, 1.0))),
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
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::Y, Vec3::Y),
    ));
}

fn update(
    mut planet_state: ResMut<PlanetState>,
    mut camera_query: Query<
        (&Transform, &mut WorldCoordinates),
        (With<Camera>, Without<Planet>, Without<PoleMarker>),
    >,
    mut planet_query: Query<&mut Transform, With<Planet>>,
    mut poles_query: Query<
        (&mut Transform, &PoleMarker),
        (With<PoleMarker>, Without<Camera>, Without<Planet>),
    >,
    mut sun_query: Query<
        (&mut DirectionalLight, &mut Transform),
        (
            With<SunLight>,
            Without<Camera>,
            Without<Planet>,
            Without<PoleMarker>,
        ),
    >,
) {
    let (camera_transform, mut world_coords) = match camera_query.single_mut() {
        Ok(res) => res,
        Err(_) => return,
    };

    // Roll the planet under the camera
    let current_pos = camera_transform.translation;
    let mut planet_transform = planet_query.single_mut().unwrap();
    if let Some(previous_pos) = planet_state.last_camera_pos {
        let delta_pos = current_pos - previous_pos;
        let delta_xz = delta_pos.xz();
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
    planet_state.last_camera_pos = Some(current_pos);

    // Compute camera’s latitude/longitude from the planet’s “up” vector
    let planet_up = planet_transform.rotation.mul_vec3(Vec3::Y);
    world_coords.latitude = planet_up.y.clamp(-1.0, 1.0).asin().to_degrees();
    world_coords.longitude = planet_up.z.atan2(planet_up.x).to_degrees();
    world_coords.altitude = camera_transform.translation.y;

    // Snap each pole onto the sphere’s surface and orient it along the normal
    let mut north_pole_pos = Vec3::ZERO;
    let mut south_pole_pos = Vec3::ZERO;
    for (mut pole_tf, pole_marker) in &mut poles_query {
        let sign = if pole_marker.is_north { 1.0 } else { -1.0 };
        let planet_center = planet_transform.translation;
        let world_normal = planet_transform
            .rotation
            .mul_vec3(Vec3::Y * sign)
            .normalize();

        // World‐space position under the sphere‐center transform
        pole_tf.translation = planet_center + world_normal * (PLANET_RADIUS + POLE_HEIGHT * 0.5);
        pole_tf.rotation = Quat::from_rotation_arc(Vec3::Y, world_normal);

        if pole_marker.is_north {
            north_pole_pos = pole_tf.translation;
        } else {
            south_pole_pos = pole_tf.translation;
        }
    }

    // --- Sun Light Control Logic ---
    let (mut sun_light, mut sun_transform) = sun_query.single_mut().unwrap();

    // Get pole to snap light to
    let dist_to_north = camera_transform.translation.distance(north_pole_pos);
    let dist_to_south = camera_transform.translation.distance(south_pole_pos);
    let target_pole_pos = if dist_to_north < dist_to_south {
        north_pole_pos
    } else {
        south_pole_pos
    };

    sun_transform.rotation = Transform::default()
        .looking_at(
            -(camera_transform.translation - target_pole_pos).normalize(),
            Vec3::Y,
        )
        .rotation;

    let min_latitude_for_light_degrees = 15.0;
    let max_latitude_for_light_degrees = 90.0;

    let current_abs_latitude = world_coords.latitude.abs();

    let intensity_factor = ((current_abs_latitude - min_latitude_for_light_degrees)
        / (max_latitude_for_light_degrees - min_latitude_for_light_degrees))
        .clamp(0.0, 1.0);

    sun_light.illuminance = lux::DIRECT_SUNLIGHT * intensity_factor;
}
