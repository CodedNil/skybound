use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
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
        if keyboard_input.pressed(KeyCode::KeyW) {
            movement += *transform.forward();
        }
        if keyboard_input.pressed(KeyCode::KeyS) {
            movement += -*transform.forward();
        }
        if keyboard_input.pressed(KeyCode::KeyA) {
            movement += -*transform.right();
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            movement += *transform.right();
        }
        if keyboard_input.pressed(KeyCode::KeyQ) {
            movement += *transform.down();
        }
        if keyboard_input.pressed(KeyCode::KeyE) {
            movement += *transform.up();
        }
        if movement.length_squared() > 0.0 {
            movement = movement.normalize();
        }
        transform.translation += movement
            * controller.speed
            * time.delta_secs()
            * (if keyboard_input.pressed(KeyCode::ShiftLeft) {
                10.0
            } else {
                1.0
            });

        // Rotation with right-click drag
        if mouse_button_input.pressed(MouseButton::Right) {
            let mut delta = Vec2::ZERO;
            for event in mouse_motion_events.read() {
                delta += event.delta;
            }
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
