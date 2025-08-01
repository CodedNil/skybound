use crate::camera::CameraController;
use bevy::{
    anti_aliasing::taa::TemporalAntiAliasing,
    camera::Exposure,
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    pbr::{CascadeShadowConfigBuilder, light_consts::lux},
    prelude::*,
};
use std::f32::consts::FRAC_PI_2;

// --- Constants ---
pub const PLANET_RADIUS: f32 = 500_000.0;
const CAMERA_RESET_THRESHOLD: f32 = 50_000.0;

// Sun Angle Configuration
const SUNSET_LATITUDE_DEG: f32 = 6.0; // The latitude at which the sun sets
const MIN_SUN_ELEVATION_DEG: f32 = -17.0; // The sun elevation at sunset latitude
const MAX_SUN_ELEVATION_DEG: f32 = 70.0; // The maximum sun elevation angle

// --- Components ---
#[derive(Resource)]
pub struct WorldCoordinates {
    pub planet_rotation: Quat,
}
impl Default for WorldCoordinates {
    fn default() -> Self {
        Self {
            planet_rotation: Quat::from_rotation_x(-FRAC_PI_2),
        }
    }
}
impl WorldCoordinates {
    /// Calculates the latitude of the world's origin (0,0,0).
    pub fn latitude(&self) -> f32 {
        let v_local = self.planet_rotation.conjugate().mul_vec3(Vec3::Y);
        v_local.y.clamp(-1.0, 1.0).asin()
    }

    /// Calculates the longitude of the world's origin (0,0,0).
    pub fn longitude(&self) -> f32 {
        let v_local = self.planet_rotation.conjugate().mul_vec3(Vec3::Y);
        if v_local.x.abs() < f32::EPSILON && v_local.z.abs() < f32::EPSILON {
            0.0
        } else {
            v_local.x.atan2(v_local.z)
        }
    }
}

#[derive(Component, Default)]
pub struct CameraCoordinates {
    last_camera_pos: Option<Vec3>,
}
impl CameraCoordinates {
    /// Calculates the rotation caused by the camera's current translation from the origin.
    fn rotation_from_translation(translation: Vec3) -> Quat {
        let delta_xz = translation.xz();
        if delta_xz.length_squared() > f32::EPSILON {
            let roll_axis = Vec3::new(-delta_xz.y, 0.0, delta_xz.x).normalize();
            let roll_angle = delta_xz.length() / PLANET_RADIUS;
            Quat::from_axis_angle(roll_axis, roll_angle)
        } else {
            Quat::IDENTITY
        }
    }

    /// Calculates the effective rotation of the planet by combining the global base rotation with the transient rotation from the camera's current position.
    pub fn planet_rotation(
        &self,
        world_coords: &WorldCoordinates,
        camera_transform: &Transform,
    ) -> Quat {
        let offset_rotation = Self::rotation_from_translation(camera_transform.translation);
        offset_rotation * world_coords.planet_rotation
    }

    /// Calculates the camera's current latitude in radians.
    pub fn latitude(&self, planet_rotation: Quat, camera_transform: &Transform) -> f32 {
        let v = Vec3::new(0.0, camera_transform.translation.y + PLANET_RADIUS, 0.0);
        let v_local = planet_rotation.conjugate().mul_vec3(v);
        let r = v_local.length();
        (v_local.y / r).clamp(-1.0, 1.0).asin()
    }

    /// Calculates the camera's current latitude in meters.
    pub fn latitude_meters(&self, latitude_rad: f32) -> f32 {
        latitude_rad * PLANET_RADIUS
    }

    /// Calculates the camera's current longitude in radians.
    pub fn longitude(&self, planet_rotation: Quat, camera_transform: &Transform) -> f32 {
        let v = Vec3::new(0.0, camera_transform.translation.y + PLANET_RADIUS, 0.0);
        let v_local = planet_rotation.conjugate().mul_vec3(v);
        if v_local.x.abs() < f32::EPSILON && v_local.z.abs() < f32::EPSILON {
            0.0
        } else {
            v_local.x.atan2(v_local.z)
        }
    }

    /// Calculates the camera's current longitude in meters.
    pub fn longitude_meters(&self, longitude_rad: f32, latitude_rad: f32) -> f32 {
        longitude_rad * PLANET_RADIUS * latitude_rad.cos()
    }

    /// Calculates the camera's current altitude in meters.
    pub fn altitude(&self, camera_transform: &Transform) -> f32 {
        camera_transform.translation.y
    }
}

#[derive(Component)]
pub struct SunLight;

// --- Plugin ---
pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldCoordinates>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

// --- Systems ---
fn setup(mut commands: Commands) {
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
        CameraCoordinates::default(),
        CameraController {
            speed: 40.0,
            sensitivity: 0.005,
            yaw: 0.0,
            pitch: 0.0,
        },
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
    mut world_coords: ResMut<WorldCoordinates>,
    mut camera_query: Query<(&mut Transform, &mut CameraCoordinates), With<Camera>>,
    mut sun_query: Query<
        (&mut Transform, &mut DirectionalLight),
        (With<SunLight>, Without<Camera>),
    >,
) {
    let (mut camera_transform, mut camera_coords) = match camera_query.single_mut() {
        Ok(res) => res,
        Err(_) => return,
    };

    // --- Camera Snapping Logic ---
    if camera_transform.translation.x.abs() > CAMERA_RESET_THRESHOLD
        || camera_transform.translation.z.abs() > CAMERA_RESET_THRESHOLD
    {
        // Calculate the rotation from the distance traveled before snapping.
        let snap_rotation =
            CameraCoordinates::rotation_from_translation(camera_transform.translation);
        // Apply this rotation to the global world coordinates.
        world_coords.planet_rotation = snap_rotation * world_coords.planet_rotation;

        // Reset the camera's position to the origin.
        camera_transform.translation.x = 0.0;
        camera_transform.translation.z = 0.0;

        // After snapping, reset the last known position to the new origin.
        camera_coords.last_camera_pos = Some(camera_transform.translation);
    }

    // --- Planet and Object Positioning ---
    // The visual rotation is the combination of the base global rotation and the camera's current offset.
    let effective_rotation = camera_coords.planet_rotation(&world_coords, &camera_transform);

    // The planet's center is always directly below the camera's XZ position, creating a "treadmill" effect.
    let planet_center = Vec3::new(
        camera_transform.translation.x,
        -PLANET_RADIUS,
        camera_transform.translation.z,
    );

    // --- Sun Logic ---
    let (mut sun_transform, mut sun_light) = sun_query.single_mut().unwrap();

    // Get the current latitude for sun calculations.
    let planet_rotation = camera_coords.planet_rotation(&world_coords, &camera_transform);
    let current_latitude = camera_coords.latitude(planet_rotation, &camera_transform);
    let latitude_abs = ((current_latitude.abs().to_degrees() - SUNSET_LATITUDE_DEG)
        / (90.0 - SUNSET_LATITUDE_DEG))
        .clamp(0.0, 1.0);

    let camera_up = (camera_transform.translation - planet_center).normalize();
    let pole_sign = if current_latitude >= 0.0 { 1.0 } else { -1.0 };
    let planet_pole_direction = effective_rotation.mul_vec3(Vec3::Y) * pole_sign;

    let sun_azimuth = (planet_pole_direction - camera_up * planet_pole_direction.dot(camera_up))
        .normalize_or((Vec3::X - camera_up * Vec3::X.dot(camera_up)).normalize());

    let desired_elevation_rad = (MIN_SUN_ELEVATION_DEG
        + (MAX_SUN_ELEVATION_DEG - MIN_SUN_ELEVATION_DEG) * latitude_abs)
        .to_radians();

    sun_transform.rotation = Quat::from_rotation_arc(
        Vec3::NEG_Z,
        -sun_azimuth * desired_elevation_rad.cos() - camera_up * desired_elevation_rad.sin(),
    );
    sun_light.illuminance = lux::DIRECT_SUNLIGHT * latitude_abs;

    // Store the camera's position for the next frame's snap check.
    camera_coords.last_camera_pos = Some(camera_transform.translation);
}
