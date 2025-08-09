use crate::camera::CameraController;
use bevy::{
    anti_aliasing::taa::TemporalAntiAliasing,
    camera::Exposure,
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    pbr::{CascadeShadowConfigBuilder, light_consts::lux},
    prelude::*,
};
use std::f32::consts::FRAC_PI_4;

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
    pub camera_offset: Vec3,
    last_camera_pos: Option<Vec3>,
}
impl Default for WorldCoordinates {
    fn default() -> Self {
        let offset = Quat::from_rotation_x(FRAC_PI_4).mul_vec3(Vec3::Y * PLANET_RADIUS);
        Self {
            camera_offset: Vec3::new(offset.x, 0.0, offset.z),
            last_camera_pos: None,
        }
    }
}
impl WorldCoordinates {
    /// Calculates the latitude at a given position.
    pub fn latitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Y);
        v_local.y.clamp(-1.0, 1.0).asin()
    }

    /// Calculates the longitude at a given position.
    pub fn longitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Y);
        if v_local.x.abs() < f32::EPSILON && v_local.z.abs() < f32::EPSILON {
            0.0
        } else {
            v_local.x.atan2(v_local.z)
        }
    }

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

    /// Calculates the effective rotation of the planet at a given position.
    pub fn planet_rotation(&self, pos: Vec3) -> Quat {
        Self::rotation_from_translation(self.camera_offset + pos)
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
}

fn update(
    mut world_coords: ResMut<WorldCoordinates>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
    mut sun_query: Query<
        (&mut Transform, &mut DirectionalLight),
        (With<SunLight>, Without<Camera>),
    >,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    // --- Camera Snapping Logic ---
    if camera_transform.translation.x.abs() > CAMERA_RESET_THRESHOLD
        || camera_transform.translation.z.abs() > CAMERA_RESET_THRESHOLD
    {
        // Reset the camera's position to the origin, first storing the offset.
        world_coords.camera_offset += Vec3::new(
            camera_transform.translation.x,
            0.0,
            camera_transform.translation.z,
        );
        camera_transform.translation.x = 0.0;
        camera_transform.translation.z = 0.0;

        // After snapping, reset the last known position to the new origin.
        world_coords.last_camera_pos = Some(camera_transform.translation);
    }

    // --- Planet and Object Positioning ---
    let effective_rotation = world_coords.planet_rotation(camera_transform.translation);

    // The planet's center is always directly below the camera's XZ position, creating a "treadmill" effect.
    // --- Sun Logic ---
    let (mut sun_transform, mut sun_light) = sun_query.single_mut().unwrap();

    // Get the current latitude for sun calculations.
    let current_latitude = world_coords.latitude(camera_transform.translation);
    let latitude_abs = ((current_latitude.abs().to_degrees() - SUNSET_LATITUDE_DEG)
        / (90.0 - SUNSET_LATITUDE_DEG))
        .clamp(0.0, 1.0);

    let pole_sign = if current_latitude >= 0.0 { 1.0 } else { -1.0 };
    let planet_pole_direction = effective_rotation.mul_vec3(Vec3::Y) * pole_sign;

    let sun_azimuth = (planet_pole_direction - Vec3::Y * planet_pole_direction.dot(Vec3::Y))
        .normalize_or((Vec3::X - Vec3::Y * Vec3::X.dot(Vec3::Y)).normalize());

    let desired_elevation_rad = (MIN_SUN_ELEVATION_DEG
        + (MAX_SUN_ELEVATION_DEG - MIN_SUN_ELEVATION_DEG) * latitude_abs)
        .to_radians();

    sun_transform.rotation = Quat::from_rotation_arc(
        Vec3::NEG_Z,
        -sun_azimuth * desired_elevation_rad.cos() - Vec3::Y * desired_elevation_rad.sin(),
    );
    sun_light.illuminance = lux::DIRECT_SUNLIGHT * latitude_abs;

    // Store the camera's position for the next frame's snap check.
    world_coords.last_camera_pos = Some(camera_transform.translation);
}
