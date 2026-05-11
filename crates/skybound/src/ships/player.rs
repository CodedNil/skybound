use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

/// Marker component for the player's ship entity.
#[derive(Component)]
pub struct PlayerShip;

/// Flight controller state attached to the ship.
#[derive(Component)]
pub struct ShipController {
    pub speed: f32,
    pub sensitivity: f32,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_ship)
            .add_systems(PreUpdate, ship_controller)
            .add_systems(PostUpdate, follow_camera_to_ship);
    }
}

fn spawn_ship(mut commands: Commands) {
    commands.spawn((
        PlayerShip,
        ShipController {
            speed: 40.0,
            sensitivity: 0.005,
            yaw: 0.0,
            pitch: 0.0,
        },
        Transform::from_xyz(0.0, 4.0, 12.0),
    ));
}

fn ship_controller(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut ShipController), With<PlayerShip>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
) {
    for (mut transform, mut controller) in &mut query {
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

        let yaw_quat = Quat::from_rotation_z(controller.yaw);
        let pitch_quat = Quat::from_rotation_x(controller.pitch + std::f32::consts::FRAC_PI_2);
        transform.rotation = yaw_quat * pitch_quat;

        for event in mouse_wheel_events.read() {
            controller.speed += event.y * 0.5 * controller.speed;
            controller.speed = controller.speed.clamp(0.1, 5000.0);
        }
    }
}

/// Smoothly follows the camera behind and above the ship.
fn follow_camera_to_ship(
    time: Res<Time>,
    ship_query: Query<&Transform, (With<PlayerShip>, Without<Camera>)>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
) {
    let Ok(ship) = ship_query.single() else {
        return;
    };
    let Ok(mut cam) = camera_query.single_mut() else {
        return;
    };

    let forward = *ship.forward();
    let up = *ship.up();
    let target = ship.translation - forward * 60.0 + up * 15.0;

    // Exponential smoothing: ~0.12 s time constant at 60 fps.
    let lag = 1.0 - (-8.0 * time.delta_secs()).exp();
    cam.translation = cam.translation.lerp(target, lag.clamp(0.0, 1.0));
    cam.look_at(ship.translation + forward * 5.0, up);
}
