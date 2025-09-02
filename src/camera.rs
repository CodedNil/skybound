use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

/// Plugin that adds the camera controller system.
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    /// Register systems required for the camera controller.
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, camera_controller);
    }
}

#[derive(Component)]
pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// Handles camera movement, rotation, and speed adjustments from input.
fn camera_controller(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut mouse_wheel_events: EventReader<MouseWheel>,
) {
    for (mut transform, mut controller) in &mut query {
        // Movement with WASDQE
        let mut movement = Vec3::ZERO;
        let mut add_dir = |key: KeyCode, v: Vec3| {
            if keyboard_input.pressed(key) {
                movement += v;
            }
        };
        add_dir(KeyCode::KeyW, *transform.forward());
        add_dir(KeyCode::KeyS, -*transform.forward());
        add_dir(KeyCode::KeyA, -*transform.right());
        add_dir(KeyCode::KeyD, *transform.right());
        add_dir(KeyCode::KeyQ, *transform.down());
        add_dir(KeyCode::KeyE, *transform.up());
        if movement.length_squared() > 0.0 {
            movement = movement.normalize();
        }

        let sprint = if keyboard_input.pressed(KeyCode::ShiftLeft) {
            10.0
        } else {
            1.0
        };
        transform.translation += movement * controller.speed * time.delta_secs() * sprint;

        // Rotation with right-click drag — accumulate mouse deltas
        if mouse_button_input.pressed(MouseButton::Right) {
            let delta = mouse_motion_events.read().fold(Vec2::ZERO, |mut acc, e| {
                acc += e.delta;
                acc
            });
            if delta != Vec2::ZERO {
                controller.yaw -= delta.x * controller.sensitivity;
                controller.pitch = (controller.pitch - delta.y * controller.sensitivity).clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.01,
                    std::f32::consts::FRAC_PI_2 - 0.01,
                );
            }
        }

        // Update camera rotation
        let yaw_quat = Quat::from_rotation_z(controller.yaw);
        let pitch_quat = Quat::from_rotation_x(controller.pitch + std::f32::consts::FRAC_PI_2);
        transform.rotation = yaw_quat * pitch_quat;

        // Speed adjustment with scroll wheel
        for event in mouse_wheel_events.read() {
            controller.speed += event.y * 0.5 * controller.speed;
            controller.speed = controller.speed.clamp(0.1, 5000.0);
        }
    }
}
