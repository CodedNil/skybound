use crate::camera::CameraController;
use bevy::{
    camera::{
        Camera3dDepthLoadOp, CameraOutputMode, ComputedCameraValues, RenderTarget,
        ScreenSpaceTransmissionQuality,
    },
    core_pipeline::bloom::Bloom,
    prelude::*,
    render::{render_resource::TextureUsages, view::Hdr},
    window::WindowRef,
};
use core::default::Default;
use std::f32::consts::FRAC_PI_4;

// --- Constants ---
pub const PLANET_RADIUS: f32 = 1_000_000.0;
const CAMERA_RESET_THRESHOLD: f32 = 50_000.0;

// --- Components ---
#[derive(Resource)]
pub struct WorldData {
    pub camera_offset: Vec2,
}
impl Default for WorldData {
    fn default() -> Self {
        let offset = Quat::from_rotation_x(FRAC_PI_4).mul_vec3(Vec3::Z * PLANET_RADIUS);
        Self {
            camera_offset: Vec2::new(
                offset.x - (offset.x % CAMERA_RESET_THRESHOLD),
                offset.y - (offset.y % CAMERA_RESET_THRESHOLD),
            ),
        }
    }
}
impl WorldData {
    /// Calculates the latitude at a given position.
    pub fn latitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Z);
        v_local.z.clamp(-1.0, 1.0).asin()
    }

    /// Calculates the longitude at a given position.
    pub fn longitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Z);
        if v_local.x.abs() < f32::EPSILON && v_local.y.abs() < f32::EPSILON {
            0.0
        } else {
            v_local.x.atan2(-v_local.y)
        }
    }

    /// Calculates the rotation caused by the camera's current translation from the origin.
    fn rotation_from_translation(translation: Vec3) -> Quat {
        let delta_xy = translation.xy();
        if delta_xy.length_squared() > f32::EPSILON {
            Quat::from_axis_angle(
                Vec3::new(delta_xy.y, -delta_xy.x, 0.0).normalize(),
                delta_xy.length() / PLANET_RADIUS,
            )
        } else {
            Quat::IDENTITY
        }
    }

    /// Calculates the effective rotation of the planet at a given position.
    pub fn planet_rotation(&self, pos: Vec3) -> Quat {
        Self::rotation_from_translation(self.camera_offset.extend(0.0) + pos)
    }
}

// --- Plugin ---
pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldData>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

// --- Systems ---
fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3d {
            depth_load_op: Camera3dDepthLoadOp::Clear(0.0),
            depth_texture_usages: TextureUsages::RENDER_ATTACHMENT.into(),
            screen_space_specular_transmission_steps: 0,
            screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::Low,
        },
        Camera {
            is_active: true,
            order: 0,
            viewport: None,
            computed: ComputedCameraValues::default(),
            target: RenderTarget::Window(WindowRef::Primary),
            output_mode: CameraOutputMode::Write {
                blend_state: None,
                clear_color: ClearColorConfig::Default,
            },
            msaa_writeback: false,
            clear_color: ClearColorConfig::Default,
            sub_camera_view: None,
        },
        Projection::Perspective(PerspectiveProjection {
            fov: FRAC_PI_4,
            near: 0.1,
            far: 1000.0,
            aspect_ratio: 1.0,
        }),
        Hdr,
        Bloom::NATURAL,
        Transform::from_xyz(0.0, 4.0, 12.0).looking_at(Vec3::Y * 4.0, Vec3::Y),
        CameraController {
            speed: 40.0,
            sensitivity: 0.005,
            yaw: 0.0,
            pitch: 0.0,
        },
    ));
}

fn update(
    mut world_coords: ResMut<WorldData>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    // --- Camera Snapping Logic ---
    if camera_transform.translation.x.abs() > CAMERA_RESET_THRESHOLD {
        let snap_amount = CAMERA_RESET_THRESHOLD * camera_transform.translation.x.signum();
        world_coords.camera_offset.x += snap_amount;
        camera_transform.translation.x -= snap_amount;
    }
    if camera_transform.translation.y.abs() > CAMERA_RESET_THRESHOLD {
        let snap_amount = CAMERA_RESET_THRESHOLD * camera_transform.translation.y.signum();
        world_coords.camera_offset.y += snap_amount;
        camera_transform.translation.y -= snap_amount;
    }
}
