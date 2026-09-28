use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
};
use skybound_gpu::ShipUniform;
use std::f32::consts::FRAC_PI_2;

use crate::camera::CameraController;

#[derive(Component)]
pub struct PlayerShip;

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct ExtractedShipData {
    pub uniform: ShipUniform,
}

fn update_ships(
    mut extracted: ResMut<ExtractedShipData>,
    ship_query: Query<&Transform, With<PlayerShip>>,
) {
    let Ok(ship) = ship_query.single() else {
        return;
    };

    extracted.uniform = ShipUniform {
        position: ship.translation.extend(1.0),
        rotation: ship.rotation.to_array().into(),
    };
}

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
            .add_systems(PostUpdate, (follow_camera_to_ship, update_ships));
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
    camera: Single<Has<CameraController>, With<Camera>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
) {
    if *camera {
        return;
    }

    for (mut transform, mut controller) in &mut query {
        let mut movement = Vec3::ZERO;
        let mut add_dir = |key: KeyCode, vector: Vec3| {
            if keyboard_input.pressed(key) {
                movement += vector;
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
            let delta = mouse_motion_events
                .read()
                .fold(Vec2::ZERO, |mut acc, event| {
                    acc += event.delta;
                    acc
                });
            if delta != Vec2::ZERO {
                controller.yaw -= delta.x * controller.sensitivity;
                controller.pitch = (controller.pitch - delta.y * controller.sensitivity)
                    .clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
            }
        }

        transform.rotation = Quat::from_rotation_z(controller.yaw)
            * Quat::from_rotation_x(controller.pitch + FRAC_PI_2);

        for event in mouse_wheel_events.read() {
            controller.speed = (controller.speed * (1.0 + event.y * 0.5)).clamp(0.1, 5000.0);
        }
    }
}

fn follow_camera_to_ship(
    time: Res<Time>,
    ship: Single<&Transform, (With<PlayerShip>, Without<Camera>)>,
    mut camera: Single<&mut Transform, (With<Camera>, Without<CameraController>)>,
) {
    let forward = *ship.forward();
    let up = *ship.up();
    let target = ship.translation - forward * 60.0 + up * 15.0;

    let lag = 1.0 - (-8.0 * time.delta_secs()).exp();
    camera.translation = camera.translation.lerp(target, lag.clamp(0.0, 1.0));
    camera.look_at(ship.translation + forward * 5.0, up);
}
