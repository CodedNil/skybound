use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

use crate::show_prepass::{ShowPrepass, ShowPrepassDepthPower};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, camera_controller)
            .add_systems(Update, choose_show_prepass_mode)
            .add_systems(Update, toggle_freecam);
    }
}

#[derive(Component)]
pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
}

/// Handles camera movement, rotation, and speed adjustments from input.
fn camera_controller(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
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
        if movement != Vec3::ZERO {
            let sprint = if keyboard_input.pressed(KeyCode::ShiftLeft) {
                10.0
            } else {
                1.0
            };
            transform.translation +=
                movement.normalize() * controller.speed * time.delta_secs() * sprint;
        }

        // Rotation with right-click drag
        if mouse_button_input.pressed(MouseButton::Right) {
            let delta = mouse_motion_events
                .read()
                .fold(Vec2::ZERO, |acc, e| acc + e.delta);

            if delta != Vec2::ZERO {
                transform.rotate_z(-delta.x * controller.sensitivity);
                let pitch_delta = -delta.y * controller.sensitivity;
                let current_pitch_sin = transform.forward().y;
                if (pitch_delta > 0.0 && current_pitch_sin < 0.99)
                    || (pitch_delta < 0.0 && current_pitch_sin > -0.99)
                {
                    transform.rotate_local_x(pitch_delta);
                }
                // Remove roll
                let final_forward = *transform.forward();
                transform.look_to(final_forward, Vec3::Z);
            }
        }

        // Speed adjustment with scroll wheel
        for event in mouse_wheel_events.read() {
            controller.speed += event.y * 0.5 * controller.speed;
            controller.speed = controller.speed.clamp(0.1, 5000.0);
        }
    }
}

fn choose_show_prepass_mode(
    mut commands: Commands,
    camera: Single<Entity, With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Digit1) {
        commands
            .entity(*camera)
            .remove::<ShowPrepass>()
            .remove::<ShowPrepassDepthPower>();
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        commands
            .entity(*camera)
            .insert(ShowPrepass::Depth)
            .insert(ShowPrepassDepthPower(0.5));
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        commands.entity(*camera).insert(ShowPrepass::Normals);
    } else if keyboard.just_pressed(KeyCode::Digit4) {
        commands.entity(*camera).insert(ShowPrepass::MotionVectors);
    }
}

fn toggle_freecam(
    mut commands: Commands,
    camera: Single<(Entity, Has<CameraController>), With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }

    let (entity, has_controller) = *camera;
    if has_controller {
        commands.entity(entity).remove::<CameraController>();
    } else {
        commands.entity(entity).insert(CameraController {
            speed: 40.0,
            sensitivity: 0.005,
        });
    }
}
